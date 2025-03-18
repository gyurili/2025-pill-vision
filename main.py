import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src import device, train_model, BASE_DIR, get_loss, predict_and_visualize_dataset
from models import DeformableDETR
from dataset import get_dataloaders, TestDataset

if __name__ == "__main__":
    CSV_PATH = BASE_DIR / "./data/image_annotations.csv"
    IMAGE_DIR = BASE_DIR / "./data/train_images"
    TEST_DIR = BASE_DIR / "./data/test_images"
    
    model = DeformableDETR(num_layers=3)
    model.to(device).float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 훈련 & 검증 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(CSV_PATH, IMAGE_DIR, batch_size=8, val_split=0.2)
    test_dataset = TestDataset(TEST_DIR)

    num_epochs = 50  # 원하는 epoch 설정
    criterion = get_loss()
    
    # 모델 학습 실행
    model, train_loss_history, val_loss_history = train_model(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs)

    # 학습 및 검증 손실 그래프 출력
    plt.plot(range(1, num_epochs + 1), train_loss_history, marker='o', linestyle='-', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_history, marker='s', linestyle='-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 예측 및 시각화 실행
    predict_and_visualize_dataset(model, test_dataset, device, class_names, threshold=0.5, num_samples=5)