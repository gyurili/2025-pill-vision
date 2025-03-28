import math
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from src import (
    device, train_model, BASE_DIR, get_loss, 
    predict_and_visualize_dataset, CLASS_NAMES,
    evaluate_map, generate_submission_csv
)
from models import DeformableDETR
from dataset import get_dataloaders, TestDataset

def custom_warmup_cosine_scheduler(
    optimizer, total_epochs, warmup_ratio=0.1, peak_ratio=0.3, warmup_pow=2.0
):
    def lr_lambda(epoch):
        if epoch < total_epochs * warmup_ratio:
            progress = epoch / (total_epochs * warmup_ratio)
            return progress**warmup_pow  # 예: pow=2 -> 느리게 시작
        elif epoch < total_epochs * peak_ratio:
            return 1.0
        else:
            progress = (epoch - total_epochs * peak_ratio) / (total_epochs * (1 - peak_ratio))
            return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# 데이터 및 모델 경로 설정
CSV_PATH = BASE_DIR / "data/image_annotations.csv"
JSON_PATH = BASE_DIR / "data/category_mapping.json"
IMAGE_DIR = BASE_DIR / "data/train_images"
TEST_DIR = BASE_DIR / "data/test_images"

if __name__ == "__main__":
    # 모델 초기화
    model = DeformableDETR(num_layers=3).to(device).float()
    # model.load_state_dict(torch.load('model_3.pth'))

    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-2
    )

    # 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(
        CSV_PATH, IMAGE_DIR, bbox_convert=True, batch_size=8, val_split=0.2
    )
    test_dataset = TestDataset(TEST_DIR)

    # 손실 함수 및 학습 설정
    num_epochs = 50
    criterion = get_loss()
    
    scheduler = custom_warmup_cosine_scheduler(
        optimizer,
        total_epochs=num_epochs,
        warmup_ratio=0.2,
        peak_ratio=0.5,
        warmup_pow=3.0  # 곡선형 증가
    )   

    # 모델 학습
    model, train_loss_history, val_loss_history = train_model(
        model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs
    )
    
    # 모델 평가 mAP@0.5
    evaluate_map(model, val_loader, class_names=CLASS_NAMES)
    
    generate_submission_csv(
        model=model,
        test_dataset=test_dataset,
        category_map_path=JSON_PATH
    )

    
    # 학습 및 검증 손실 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_loss_history, marker="o", linestyle="-", label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_loss_history, marker="s", linestyle="-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 예측 및 시각화 실행
    predict_and_visualize_dataset(
        model, test_dataset, CLASS_NAMES, threshold=0.5, num_samples=5
    )