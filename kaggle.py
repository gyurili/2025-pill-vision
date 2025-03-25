import os
import torch
import pandas as pd
import json
from torchvision import transforms
from torchvision.ops import nms  # NMS 함수 추가
from PIL import Image
from models.faster_rcnn import get_faster_rcnn_model

# 경로 설정
TEST_IMAGE_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"
OUTPUT_CSV = "/content/2025-health-vision/data/kaggle/result.csv"
CHECKPOINT_PATH = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트/Adam_0.0001/faster_rcnn_epoch6.pth"
CATEGORY_MAPPING_PATH = "/content/2025-health-vision/data/category_mapping.json"

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 82
model = get_faster_rcnn_model(num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# JSON에서 category_id 매핑 로드
with open(CATEGORY_MAPPING_PATH, "r", encoding="utf-8") as f:
    category_mapping = json.load(f)

reverse_mapping = {v: int(k) for k, v in category_mapping.items()}


transform = transforms.Compose([transforms.ToTensor()])
image_files = sorted(os.listdir(TEST_IMAGE_DIR))
results = []
annotation_id = 1

for img_file in image_files:
    image_path = os.path.join(TEST_IMAGE_DIR, img_file)
    image_id = int(os.path.splitext(img_file)[0])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # NMS로 중복 예측 제거
    keep = nms(predictions["boxes"], predictions["scores"], iou_threshold=0.5)  # IOU threshold를 0.5로 설정

    # category_id별 최고 score만 유지
    best_predictions = {}

    # 결과 저장
    for i in keep:
        bbox = predictions["boxes"][i].cpu().numpy()
        model_category_id = int(predictions["labels"][i].cpu().numpy())
        score = float(predictions["scores"][i].cpu().numpy())

        if score < 0.3:
            continue

        if model_category_id in reverse_mapping:
            category_id = reverse_mapping[model_category_id]
        else:
            continue

        if category_id not in best_predictions or best_predictions[category_id]["score"] < score:
            best_predictions[category_id] = {
                "bbox": bbox,
                "score": round(score, 2)
            }

    for category_id, data in best_predictions.items():
        bbox = data["bbox"]
        score = data["score"]

        results.append([
            annotation_id, image_id, category_id,  # category_id 변경됨
            int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), score
        ])
        annotation_id += 1

# CSV 저장
df = pd.DataFrame(results, columns=["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"예측 결과가 {OUTPUT_CSV}에 저장되었습니다.")
