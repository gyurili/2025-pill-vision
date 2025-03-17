import torch
import os
import cv2
import numpy as np
from models.faster_rcnn import get_faster_rcnn_model
from dataset.pill_dataset import TestDataset
from src.visualization import visualize_sample

def load_model(model_path, num_classes, device="cuda"):
    """
    저장된 Faster R-CNN 모델을 불러오는 함수
    
    Args:
        model_path (str): 저장된 모델의 경로
        num_classes (int): 모델의 클래스 수
        device (str): 사용할 디바이스 ("cuda" or "cpu")

    Returns:
        model (torch.nn.Module): 로드된 모델
    """
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, test_loader, device="cuda"):
    """
    Faster R-CNN 모델을 사용하여 테스트 이미지에서 객체 탐지 수행
    
    Args:
        model (torch.nn.Module): 학습된 Faster R-CNN 모델
        test_loader (DataLoader): 테스트 데이터로더
        device (str): 사용할 디바이스 ("cuda" or "cpu")
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, file_names in test_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for i, pred in enumerate(predictions):
                boxes = pred["boxes"].cpu().numpy().astype(int)
                labels = pred["labels"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                
                img_path = os.path.join(TEST_IMAGES, file_names[i])
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 신뢰도가 0.5 이상인 경우만 표시
                filtered_boxes = []
                filtered_labels = []
                for j, score in enumerate(scores):
                    if score > 0.5:
                        filtered_boxes.append(boxes[j])
                        filtered_labels.append(labels[j])
                
                target = {"boxes": torch.tensor(filtered_boxes), "labels": torch.tensor(filtered_labels)}
                visualize_sample(images[i], image, target, class_id=True)
