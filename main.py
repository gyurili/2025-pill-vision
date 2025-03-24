import torch
from models.faster_rcnn import get_faster_rcnn_model
from dataset.data_loader import get_dataloaders
from src.train import train
import os
import json  

if __name__ == "__main__":
    BASE_PATH = "/content/2025-health-vision/data"
    CSV_PATH = os.path.join(BASE_PATH, "image_annotations.csv")
    MAPPING_PATH = os.path.join(BASE_PATH, "category_mapping.json")
    TRAIN_IMAGES = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/train_images"

    # category_mapping 로드
    with open(MAPPING_PATH, "r") as f:
        category_mapping = json.load(f)

    num_classes = len(category_mapping)  # 클래스 개수
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMAGES)

    # 하이퍼파라미터 튜닝 실험
    optimizers = ["Adam", "SGD"]
    learning_rates = [0.001, 0.0005, 0.0001]

    for optimizer in optimizers:
        for lr in learning_rates:
            print(f"\n🔹 실험 시작: Optimizer={optimizer}, LR={lr}\n")
            model = get_faster_rcnn_model(num_classes)
            train(model, train_loader, val_loader, num_epochs=5, optimizer_name=optimizer, lr=lr, device=device)
