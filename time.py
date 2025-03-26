# test 디렉토리에서 100개의 이미지를 예측하고 FPS 계산
import torch
import os
import time
from models.faster_rcnn import get_faster_rcnn_model
from torchvision import transforms
from PIL import Image
from pathlib import Path

CHECKPOINT_PATH = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트/Adam_0.0001/faster_rcnn_epoch6.pth"
TEST_IMAGE_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"

# 모델 로드
def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 82
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# 이미지 예측
def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions[0]

if __name__ == "__main__":
    print(f"체크포인트 경로: {CHECKPOINT_PATH}")

    # 모델 로드
    model, device = load_model(CHECKPOINT_PATH)
    print("모델이 로드되었습니다.")
    
    total_time = 0
    num_images = 100
    count = 0

    image_files = sorted(os.listdir(TEST_IMAGE_DIR))
    image_files = image_files[:num_images]  # 100개 이미지

    start_time = time.time()  # 시작 시간

    for img_file in image_files:
        image_path = os.path.join(TEST_IMAGE_DIR, img_file)
        predictions = predict_image(model, image_path, device)
        
        # 예측 결과 출력
        #print(f"예측 결과 ({img_file}): {predictions}")

        count += 1

    # 총 시간
    total_time = time.time() - start_time

    # FPS 계산
    fps = num_images / total_time
    print(f"\n예측에 걸린 시간: {total_time:.2f}초")
    print(f"초당 프레임 수 (FPS): {fps:.2f}")
