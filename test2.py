import torch
import os
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from models.faster_rcnn import get_faster_rcnn_model
from dataset import TestDataset

CHECKPOINT_DIR = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트"
SAVE_DIR = "/content/2025-health-vision/data/finish"
CATEGORY_NAME_MAPPING_PATH = "/content/2025-health-vision/data/category_name_mapping.json"
FONT_PATH = "/content/2025-health-vision/NanumGothic-Regular.ttf"

os.makedirs(SAVE_DIR, exist_ok=True)

# 한글 폰트 로드
try:
    font = ImageFont.truetype(FONT_PATH, 20)
except:
    font = ImageFont.load_default()

def find_latest_checkpoint():
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("faster_rcnn_epoch") and f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError("체크포인트가 없습니다.")

    checkpoint_files.sort(key=lambda x: int(x.split("epoch")[1].split(".pth")[0]), reverse=True)
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
    
    return latest_checkpoint

def load_category_names():
    """ JSON 파일에서 카테고리 이름 매핑 로드 """
    with open(CATEGORY_NAME_MAPPING_PATH, "r", encoding="utf-8") as f:
        category_name_mapping = json.load(f)
    return category_name_mapping

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    TEST_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"

    MODEL_PATH = find_latest_checkpoint()
    print(f"가장 최신 모델 로드: {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 82
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("모델이 로드되었습니다.")

    # 카테고리 이름 매핑 로드
    category_name_mapping = load_category_names()

    test_dataset = TestDataset(TEST_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    max_images = 5
    count = 0

    for images, file_names in test_loader:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            predictions = model(images)

        pred = predictions[0]
        keep = pred["scores"] > 0.3
        pred["boxes"] = pred["boxes"][keep]
        pred["labels"] = pred["labels"][keep]
        pred["scores"] = pred["scores"][keep]

        print(f"예측 결과 ({file_names[0]}): {pred}")

        def save_visualization(image, prediction, file_name):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = ((image * 0.5) + 0.5) * 255
            image = image.astype("uint8")

            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)

            for box, label in zip(prediction["boxes"], prediction["labels"]):
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                label_id = str(int(label.item()))  # 모델 라벨을 문자열 ID로 변환
                label_name = category_name_mapping.get(label_id, "알 수 없음")

                # 바운딩 박스 그리기
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

                # 한글 라벨 추가
                draw.text((x_min, y_min - 25), label_name, font=font, fill="lime")

            save_path = os.path.join(SAVE_DIR, f"{file_name}.png")
            image.save(save_path)
            print(f"이미지 저장됨: {save_path}")

        save_visualization(images[0], pred, file_names[0])

        count += 1
        if count >= max_images:
            break