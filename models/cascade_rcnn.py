import os
import torch
import logging
from detectron2.engine import DefaultTrainer
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.logger import setup_logger
from detectron2.engine.hooks import PeriodicCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

# 로거 설정
logger = setup_logger(name="detectron2.custom_trainer")

class CustomTrainer(DefaultTrainer):
    """
    커스텀 트레이너 클래스
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._should_stop_early = False
    
    def stop_early(self):
        """
        학습을 조기에 중단하도록 플래그 설정
        """
        self._should_stop_early = True
        logger.info("Early stopping 요청됨. 최종 모델 저장 중...")
        
        # 최종 모델 저장
        checkpointer = DetectionCheckpointer(
            self.model, save_dir=self.cfg.OUTPUT_DIR
        )
        checkpointer.save("model_final")
        
        logger.info("최종 모델 저장 완료. 현재 iteration 완료 후 학습이 중단됩니다.")
    
    def run_step(self):
        """
        각 학습 스텝 실행 전에 조기 종료 플래그 확인
        """
        if self._should_stop_early:
            # 학습 루프를 탈출하도록 최대 iteration을 현재 값으로 설정
            self.max_iter = self.iter
            logger.info(f"Early stopping에 의해 학습 중단 (iteration {self.iter})")
            return
        
        # 기본 학습 스텝 실행
        super().run_step()
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        기본 SGD 대신 Adam 옵티마이저 사용
        
        Args:
            cfg: 설정 객체
            model: 모델
            
        Returns:
            optimizer: 옵티마이저
        """
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            # 가중치 감소(weight decay) 파라미터 설정
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "norm" in key or "bias" in key:
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            params += [{"params": [value], "lr": cfg.SOLVER.BASE_LR, "weight_decay": weight_decay}]
        
        # AdamW 사용
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
        
        return optimizer

def disable_periodic_checkpoints(trainer):
    """
    기본 체크포인트 저장 비활성화
    
    Args:
        trainer: 트레이너 객체
        
    Returns:
        disabled: 비활성화 성공 여부
    """
    for hook in trainer._hooks:
        if isinstance(hook, PeriodicCheckpointer):
            # 비활성화 또는 간격 조정
            hook.period = 999999
            logger.info("주기적 체크포인트 저장이 비활성화됨")
            return True
    return False

def setup_writers(trainer, output_dir):
    """
    이벤트 기록 설정
    
    Args:
        trainer: 트레이너 객체
        output_dir: 출력 디렉토리
    """
    trainer._writers = (
        CommonMetricPrinter(20),  # 콘솔 출력 (20 iteration마다)
        JSONWriter(os.path.join(output_dir, "metrics.json")),  # JSON 기록
        TensorboardXWriter(output_dir),  # TensorBoard 기록
    )