import os
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def create_predictor(cfg):
    """
    모델 예측기 생성
    
    Args:
        cfg: 모델 설정
        
    Returns:
        predictor: 예측기 객체
    """
    return DefaultPredictor(cfg)

def run_inference_on_image(predictor, image_path):
    """
    단일 이미지에 대한 추론 수행
    
    Args:
        predictor: 예측기 객체
        image_path: 이미지 경로
        
    Returns:
        im: 원본 이미지
        outputs: 모델 출력 결과
    """
    # 이미지 불러오기 (OpenCV 사용, BGR 포맷)
    im = cv2.imread(image_path)
    
    # 추론 수행
    outputs = predictor(im)
    
    return im, outputs

def run_inference_on_directory(predictor, image_dir, sample_count=None):
    """
    이미지 디렉토리에 대한 추론 수행
    
    Args:
        predictor: 예측기 객체
        image_dir: 이미지 디렉토리 경로
        sample_count: 샘플링할 이미지 개수 (기본값: None - 전체)
        
    Returns:
        results: 이미지 경로와 예측 결과의 딕셔너리
    """
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 샘플링
    if sample_count and sample_count < len(image_files):
        image_files = random.sample(image_files, sample_count)
    
    results = {}
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        im, outputs = run_inference_on_image(predictor, img_path)
        results[img_path] = (im, outputs)
    
    return results

def evaluate_model(cfg, dataset_name):
    """
    데이터셋에 대한 모델 평가 수행
    
    Args:
        cfg: 모델 설정
        dataset_name: 데이터셋 이름
        
    Returns:
        results: 평가 결과
    """
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    
    # 평가 실행
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    return results