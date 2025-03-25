import os
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.ops import nms  # NMS 함수 추가
from PIL import Image
from models.faster_rcnn import get_faster_rcnn_model

# 경로 설정
TEST_IMAGE_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"
OUTPUT_DIR = "/content/2025-health-vision/data/kaggle"  # 저장할 폴더
CHECKPOINT_PATH = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트/Adam_0.0001/faster_rcnn_epoch6.pth"
CATEGORY_MAPPING_PATH = "/content/2025-health-vision/data/category_mapping.json"

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 82
model = get_faster_rcnn_model(num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

with open(CATEGORY_MAPPING_PATH, "r", encoding="utf-8") as f:
    category_mapping = json.load(f)

reverse_mapping = {v: int(k) for k, v in category_mapping.items()}

transform = transforms.Compose([transforms.ToTensor()])
image_files = sorted(os.listdir(TEST_IMAGE_DIR))

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_file in image_files:
    image_path = os.path.join(TEST_IMAGE_DIR, img_file)
    image_id = int(os.path.splitext(img_file)[0])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 원본 이미지 크기 가져오기
    original_width, original_height = image.size

    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # confidence 0.3 이상인 bbox만 필터링
    high_confidence_indices = predictions["scores"] >= 0.3
    filtered_boxes = predictions["boxes"][high_confidence_indices].cpu().numpy()
    filtered_scores = predictions["scores"][high_confidence_indices].cpu().numpy()

    if len(filtered_boxes) == 0:
        print(f"{img_file}: 0.3 이상인 bbox 없음")
        continue

    # NMS로로 중복된 박스 제거
    keep = nms(torch.tensor(filtered_boxes), torch.tensor(filtered_scores), iou_threshold=0.5)
    filtered_boxes = filtered_boxes[keep]
    filtered_scores = filtered_scores[keep]

    # 시각화
    fig, ax = plt.subplots(1, figsize=(8, 10))
    ax.imshow(image)

    for bbox, score in zip(filtered_boxes, filtered_scores):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 5, f"{score:.2f}", 
            fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5)
        )  # bbox 위에 score 표시

    save_path = os.path.join(OUTPUT_DIR, f"{image_id}_bbox.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    print(f"저장 완료: {save_path}")
