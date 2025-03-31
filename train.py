#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from datetime import datetime

# 모듈 임포트
from config.config import setup_cfg, get_medication_classes
from data.dataset import register_datasets
from models.custom_trainer import CustomTrainer, disable_periodic_checkpoints, setup_writers
from utils.hooks import EvaluationHook, BestCheckpointHook, AdaptiveLRScheduler, EarlyStoppingHook
from utils.logger import setup_logger

def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description="약물 분류 모델 학습")
    parser.add_argument("--data-root", type=str, required=True, help="데이터 루트 경로")
    parser.add_argument("--image-data-root", type=str, required=True, help="이미지 데이터 경로")
    parser.add_argument("--output-dir", type=str, default="output", help="출력 디렉토리")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 이어서 학습")
    parser.add_argument("--checkpoint", type=str, default=None, help="사용할 체크포인트 경로")
    parser.add_argument("--max-iter", type=int, default=20000, help="최대 반복 횟수")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.0001, help="학습률")
    parser.add_argument("--eval-period", type=int, default=500, help="평가 주기 (iteration)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    return parser.parse_args()

def main():
    """
    메인 함수
    """
    # 명령행 인자 파싱
    args = parse_args()
    
    # 출력 경로 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger("medicine_detection", output_path)
    logger.info("약물 분류 모델 학습 시작")
    logger.info(f"설정: {args}")
    
    # 데이터셋 등록
    register_datasets(args.image_data_root)
    
    # 체크포인트 경로
    checkpoint_path = None
    if args.resume and args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.resume:
        # 체크포인트 자동 탐색
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pth')]
        if checkpoint_files:
            if "model_best.pth" in checkpoint_files:
                checkpoint_path = os.path.join(args.output_dir, "model_best.pth")
            else:
                checkpoint_path = os.path.join(args.output_dir, "model_final.pth")
            logger.info(f"체크포인트 자동 탐색: {checkpoint_path}")
    
    # 설정 생성
    cfg = setup_cfg(
        args.data_root,
        args.image_data_root,
        output_path,
        args.resume,
        checkpoint_path
    )
    
    # 커스텀 파라미터 설정
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    
    # 트레이너 생성
    trainer = CustomTrainer(cfg)
    
    # 이어서 학습할 경우
    if args.resume and checkpoint_path:
        logger.info(f"체크포인트에서 이어서 학습: {checkpoint_path}")
        trainer.resume_or_load(resume=True)
    else:
        logger.info("처음부터 학습 시작")
        trainer.resume_or_load(resume=False)
    
    # 기본 체크포인트 저장 비활성화
    disabled = disable_periodic_checkpoints(trainer)
    if not disabled:
        logger.warning("PeriodicCheckpointer 훅을 찾을 수 없습니다. 자동 체크포인트 저장이 여전히 활성화될 수 있습니다.")
    
    # 평가 훅 생성
    eval_hook = EvaluationHook(
        eval_period=args.eval_period,
        eval_dataset="my_dataset_val",
        output_dir=output_path
    )
    
    # 최고 성능 체크포인트 저장 훅
    best_model_hook = BestCheckpointHook(
        eval_hook=eval_hook,
        output_dir=output_path
    )
    
    # 적응형 학습률 스케줄러 훅
    lr_scheduler_hook = AdaptiveLRScheduler(
        eval_hook=eval_hook,
        patience=3,  # 3번 연속 성능 정체 시 학습률 감소
        lr_factor=0.1  # 학습률을 10%로 감소
    )
    
    # Early Stopping 훅 추가
    early_stopping_hook = EarlyStoppingHook(
        eval_hook=eval_hook,
        patience=10,  # 10번 연속 성능 정체 시 학습 종료
        min_delta=0.001  # 0.1% 이상 향상되어야 향상으로 간주
    )
    
    # 훅 등록 (순서 중요: 평가 훅이 먼저 실행되어야 함)
    trainer.register_hooks([eval_hook, best_model_hook, lr_scheduler_hook, early_stopping_hook])
    
    # TensorBoard 이벤트 기록 설정
    setup_writers(trainer, output_path)
    
    # 학습 실행
    logger.info("학습 시작")
    trainer.train()
    logger.info("학습 완료")

if __name__ == "__main__":
    main()