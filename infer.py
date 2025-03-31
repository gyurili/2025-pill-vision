#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import matplotlib.pyplot as plt
import logging

# 모듈 임포트
from config.config import get_medication_classes, setup_test_cfg
from data.dataset import setup_test_metadata
from evaluation.inference import create_predictor, run_inference_on_directory, run_inference_on_image
from utils.visualization import visualize_detection_korean, display_image
from utils.logger import setup_logger

def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description="약물 분류 모델 추론")
    parser.add_argument("--output-dir", type=str, required=True, help="모델 출력 디렉토리")
    parser.add_argument("--image-path", type=str, help="추론할 이미지 경로 (단일 이미지)")
    parser.add_argument("--image-dir", type=str, help="추론할 이미지 디렉토리 (다중 이미지)")
    parser.add_argument("--sample-count", type=int, default=30, help="샘플링할 이미지 수")
    parser.add_argument("--font-path", type=str, required=True, help="한글 폰트 경로")
    parser.add_argument("--weight-path", type=str, help="사용할 모델 가중치 경로 (기본값: model_final.pth)")
    parser.add_argument("--confidence", type=float, default=0.5, help="객체 감지 신뢰도 임계값")
    
    return parser.parse_args()

def main():
    """
    메인 함수
    """
    # 명령행 인자 파싱
    args = parse_args()
    
    # 로거 설정
    logger = setup_logger("medicine_detection_inference", args.output_dir)
    logger.info("약물 분류 모델 추론 시작")
    
    # 클래스 목록 가져오기
    classes = get_medication_classes()
    
    # 테스트 메타데이터 설정
    metadata_name = setup_test_metadata(classes)
    
    # 모델 가중치 경로 설정
    weight_path = args.weight_path
    if not weight_path:
        weight_path = os.path.join(args.output_dir, "model_final.pth")
        # model_best.pth 존재하면 이것을 사용
        best_model_path = os.path.join(args.output_dir, "model_best.pth")
        if os.path.exists(best_model_path):
            weight_path = best_model_path
    
    logger.info(f"모델 가중치 사용: {weight_path}")
    
    # 테스트 설정 생성
    cfg = setup_test_cfg(args.output_dir, classes, weight_path)
    
    # 예측기 생성
    predictor = create_predictor(cfg)
    
    # 단일 이미지 추론
    if args.image_path:
        logger.info(f"단일 이미지 추론: {args.image_path}")
        
        im, outputs = run_inference_on_image(predictor, args.image_path)
        
        # 시각화
        result_img = visualize_detection_korean(
            im,
            outputs,
            class_names=classes,
            font_path=args.font_path
        )
        
        # 결과 출력
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.title(f"예측 결과: {os.path.basename(args.image_path)}")
        plt.show()
    
    # 디렉토리 추론
    elif args.image_dir:
        logger.info(f"디렉토리 추론: {args.image_dir}, 샘플 수: {args.sample_count}")
        
        # 추론 실행
        results = run_inference_on_directory(
            predictor,
            args.image_dir,
            sample_count=args.sample_count
        )
        
        # 결과 시각화
        for img_path, (im, outputs) in results.items():
            # 시각화
            result_img = visualize_detection_korean(
                im,
                outputs,
                class_names=classes,
                font_path=args.font_path
            )
            
            # 결과 출력
            plt.figure(figsize=(12, 8))
            plt.imshow(result_img)
            plt.axis('off')
            plt.title(f"예측 결과: {os.path.basename(img_path)}")
            plt.show()
    
    else:
        logger.error("--image-path 또는 --image-dir 중 하나는 반드시 지정해야 합니다.")
        return
    
    logger.info("추론 완료")

if __name__ == "__main__":
    main()