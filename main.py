import torch
from models.faster_rcnn import get_faster_rcnn_model
from dataset.data_loader import get_dataloaders
from src.train import train
from test2 import load_model, predict
import os
import json  # JSON 파일 로드를 위해 추가

if __name__ == "__main__":
    BASE_PATH = "/content/2025-health-vision/data"
    CSV_PATH = os.path.join(BASE_PATH, "image_annotations_fixed.csv")
    MAPPING_PATH = os.path.join(BASE_PATH, "category_mapping.json")
    TRAIN_IMAGES = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/train_images"
    TEST_IMAGES = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"

    # category_mapping 로드
    with open(MAPPING_PATH, "r") as f:
        category_mapping = json.load(f)

    # num_classes를 매핑된 클래스 개수로 설정
    num_classes = len(category_mapping)  # 매핑된 클래스 개수

    # 모델 저장 경로
    model_save_path = "models/faster_rcnn_epoch10.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로더 준비
    train_loader, val_loader = get_dataloaders(CSV_PATH, TRAIN_IMAGES)

    # 모델 학습
    model = get_faster_rcnn_model(num_classes)
    train(model, train_loader, val_loader, num_epochs=10, device=device)

    # 모델 저장
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")

    # 모델 로드 후 예측
    test_dataset = TestDataset(TEST_IMAGES)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(model_save_path, num_classes, device)
    predict(model, test_loader)
