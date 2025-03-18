import torch
import os
from pathlib import Path
from models.faster_rcnn import get_faster_rcnn_model
from dataset import TestDataset
from src import visualize_sample

if __name__ == "__main__":
    """테스트 스크립트: 모델 로드 및 예측"""

    # 현재 파일이 위치한 디렉토리를 기준으로 경로 설정
    BASE_DIR = Path(__file__).resolve().parent
    TEST_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"
    MODEL_PATH = os.path.join(BASE_DIR, "models/faster_rcnn_epoch5.pth")

    # 모델 불러오기
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10  # 학습할 때 사용한 클래스 개수와 맞춰야 함!
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print(f"✅ 모델이 로드되었습니다: {MODEL_PATH}")

    # 테스트 데이터 로드
    test_dataset = TestDataset(TEST_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 테스트 수행
    for images, file_names in test_loader:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            predictions = model(images)

        print(f"🔍 예측 결과 ({file_names[0]}): {predictions[0]}")

        # 시각화
        visualize_sample(images[0], predictions[0])
        break  # 첫 번째 이미지만 테스트 후 종료
