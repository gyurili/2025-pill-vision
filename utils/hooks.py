import os
import copy
import logging
from detectron2.engine import HookBase
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.logger import setup_logger

# 로거 설정
logger = setup_logger(name="detectron2.custom_hooks")

class EvaluationHook(HookBase):
    """
    모델 평가 훅: 주기적으로 모델을 평가하고 결과를 기록
    """
    def __init__(self, eval_period, eval_dataset, output_dir):
        """
        Args:
            eval_period: 평가 주기 (iteration)
            eval_dataset: 평가 데이터셋 이름
            output_dir: 출력 디렉토리
        """
        self._period = eval_period
        self._dataset = eval_dataset
        self._output_dir = output_dir
        self.latest_results = None  # 다른 훅들이 참조할 최신 평가 결과
        self.latest_ap = -1.0       # 최신 AP 점수
        self._writer = SummaryWriter(log_dir=output_dir)

    def after_step(self):
        """매 iteration 후 실행되는 메서드"""
        if (self.trainer.iter + 1) % self._period == 0 or self.trainer.iter + 1 == self.trainer.max_iter:
            # 로그 메시지
            logger.info(f"Iteration {self.trainer.iter + 1}: 모델 평가 중...")

            # 평가 도구 설정
            evaluator = COCOEvaluator(self._dataset, self.trainer.cfg, False, output_dir=self._output_dir)
            val_loader = build_detection_test_loader(self.trainer.cfg, self._dataset)

            # 모델 평가 실행
            self.latest_results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

            # AP 점수 추출
            self.latest_ap = self.latest_results["bbox"]["AP"]
            logger.info(f"Iteration {self.trainer.iter + 1}: mAP = {self.latest_ap:.4f}")

            # TensorBoard 로깅
            if self.latest_results is not None:
                for k, v in self.latest_results["bbox"].items():
                    if isinstance(v, (int, float)):
                        self._writer.add_scalar(f"validation_bbox/{k}", v, self.trainer.iter)
                self._writer.add_scalar("validation/mAP", self.latest_ap, self.trainer.iter)
                self._writer.flush()

            # 결과 리턴하여 다른 훅들이 활용할 수 있도록 함
            return self.latest_results

        return None

    def after_train(self):
        """학습 종료 후 실행되는 메서드"""
        # 훅이 종료될 때 SummaryWriter 닫기
        if hasattr(self, '_writer'):
            self._writer.close()

class BestCheckpointHook(HookBase):
    """
    최고 성능 모델 저장 훅
    """
    def __init__(self, eval_hook, output_dir):
        """
        Args:
            eval_hook: 평가 훅 객체
            output_dir: 출력 디렉토리
        """
        self._eval_hook = eval_hook
        self._output_dir = output_dir
        self._best_ap = -1.0
        self._last_processed_iter = -1

    def after_step(self):
        """매 iteration 후 실행되는 메서드"""
        if (self._eval_hook.latest_results is not None and
            self.trainer.iter != self._last_processed_iter and
            (self.trainer.iter % self._eval_hook._period == 0 or
            self.trainer.iter + 1 == self.trainer.max_iter)):

            ap = self._eval_hook.latest_ap
            logger.info(f"[Best Model 평가] 현재 iteration: {self.trainer.iter}, 현재 AP: {ap:.4f}, 최고 AP: {self._best_ap:.4f}")

            # 최고 AP 갱신 시
            if ap > self._best_ap:
                self._best_ap = ap
                logger.info(f"[Best Model] 새로운 최고 성능 달성! mAP = {ap:.4f}, 모델 저장 중...")

                # 현재 모델의 복사본 생성
                best_model_state = copy.deepcopy(self.trainer.model.state_dict())
                
                # 상태 딕셔너리 준비
                additional_state = {
                    "iteration": self.trainer.iter,
                    "optimizer": copy.deepcopy(self.trainer.optimizer.state_dict()),
                    "scheduler": copy.deepcopy(self.trainer.scheduler.state_dict()),
                    "best_ap": ap
                }
                
                # 체크포인터로 저장
                checkpointer = DetectionCheckpointer(
                    self.trainer.model,
                    save_dir=self._output_dir
                )
                checkpointer.save("model_best", additional_state=additional_state)
                
                # 모델 저장 완료 로그
                logger.info(f"[Best Model] 저장 완료 (AP={ap:.4f}, iteration={self.trainer.iter})")

            # 처리한 iteration 기록
            self._last_processed_iter = self.trainer.iter

class AdaptiveLRScheduler(HookBase):
    """
    적응형 학습률 스케줄러 훅
    """
    def __init__(self, eval_hook, patience=5, lr_factor=0.1):
        """
        Args:
            eval_hook: 평가 훅 객체
            patience: 성능 정체 허용 횟수
            lr_factor: 학습률 감소 비율
        """
        self._eval_hook = eval_hook
        self._patience = patience
        self._lr_factor = lr_factor
        self._best_ap = -1.0
        self._stagnant_epochs = 0
        self._last_processed_iter = -1  # 마지막으로 처리한 iteration 기록

    def after_step(self):
        """매 iteration 후 실행되는 메서드"""
        if (self._eval_hook.latest_results is not None and
            self.trainer.iter != self._last_processed_iter and
            (self.trainer.iter % self._eval_hook._period == 0 or
             self.trainer.iter + 1 == self.trainer.max_iter)):

            ap = self._eval_hook.latest_ap
            logger.info(f"[LR Scheduler] 현재 iteration: {self.trainer.iter}, 현재 AP: {ap:.4f}")

            # 성능 향상 여부 확인
            if ap > self._best_ap:
                logger.info(f"[LR Scheduler] 성능 향상: {self._best_ap:.4f} -> {ap:.4f}")
                self._best_ap = ap
                self._stagnant_epochs = 0
            else:
                self._stagnant_epochs += 1
                logger.info(f"[LR Scheduler] 성능 정체 {self._stagnant_epochs}/{self._patience}회 (현재: {ap:.4f}, 최고: {self._best_ap:.4f})")

                # patience 초과시 학습률 감소
                if self._stagnant_epochs >= self._patience:
                    old_lr = self.trainer.optimizer.param_groups[0]["lr"]
                    new_lr = old_lr * self._lr_factor
                    logger.info(f"[LR Scheduler] 학습률 감소: {old_lr:.6f} -> {new_lr:.6f}")

                    # 학습률 업데이트 (optimizer와 scheduler 모두)
                    for param_group in self.trainer.optimizer.param_groups:
                        param_group["lr"] = new_lr

                    # 중요: 스케줄러의 base_lr도 업데이트
                    if hasattr(self.trainer.scheduler, "base_lrs"):
                        self.trainer.scheduler.base_lrs = [new_lr for _ in self.trainer.scheduler.base_lrs]

                    # 카운터 리셋
                    self._stagnant_epochs = 0

            # 처리한 iteration 기록
            self._last_processed_iter = self.trainer.iter

class EarlyStoppingHook(HookBase):
    """
    조기 종료 훅: 일정 횟수 이상 성능 향상이 없으면 학습을 조기 종료
    """
    def __init__(self, eval_hook, patience=10, min_delta=0.001):
        """
        Args:
            eval_hook: 평가 훅 객체
            patience: 성능 정체 허용 횟수
            min_delta: 성능 향상으로 간주할 최소 차이
        """
        self._eval_hook = eval_hook
        self._patience = patience
        self._min_delta = min_delta
        self._best_ap = -1.0
        self._stagnant_counts = 0
        self._last_processed_iter = -1

    def after_step(self):
        """매 iteration 후 실행되는 메서드"""
        # 새로운 평가 결과가 있을 때만 처리
        if (self._eval_hook.latest_results is not None and
            self.trainer.iter != self._last_processed_iter and
            (self.trainer.iter % self._eval_hook._period == 0 or
             self.trainer.iter + 1 == self.trainer.max_iter)):

            ap = self._eval_hook.latest_ap
            logger.info(f"[Early Stopping] 현재 iteration: {self.trainer.iter}, 현재 AP: {ap:.4f}, 최고 AP: {self._best_ap:.4f}")

            # 성능 향상 확인
            if ap > (self._best_ap + self._min_delta):
                logger.info(f"[Early Stopping] 성능 향상: {self._best_ap:.4f} -> {ap:.4f}")
                self._best_ap = ap
                self._stagnant_counts = 0
            else:
                self._stagnant_counts += 1
                logger.info(f"[Early Stopping] 성능 정체 {self._stagnant_counts}/{self._patience}회")

                # patience 초과시 학습 조기 종료
                if self._stagnant_counts >= self._patience:
                    logger.info(f"[Early Stopping] {self._patience}회 연속 성능 향상 없음. 학습 조기 종료!")

                    # 커스텀 트레이너의 stop_early 메서드 호출
                    if hasattr(self.trainer, 'stop_early'):
                        self.trainer.stop_early()
                    else:
                        # 기본 방식으로 종료 시도 (DefaultTrainer 사용 시)
                        self.trainer.max_iter = self.trainer.iter
                        logger.warning("커스텀 트레이너가 아니므로 즉시 종료가 보장되지 않을 수 있습니다.")

            # 처리한 iteration 기록
            self._last_processed_iter = self.trainer.iter