import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from models.faster_rcnn import get_faster_rcnn_model

# 경로 설정
TEST_IMAGE_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/test_images"
OUTPUT_CSV = "/content/drive/MyDrive/코드잇/초급 프로젝트/캐글 예측 결과/result.csv"
CHECKPOINT_PATH = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트/Adam_0.0001/faster_rcnn_epoch5.pth"  # 가장 성능 좋은 모델

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 82
model = get_faster_rcnn_model(num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor()
])

# 테스트 이미지 리스트 가져오기
image_files = sorted(os.listdir(TEST_IMAGE_DIR))
results = []
annotation_id = 1

for img_file in image_files:
    image_path = os.path.join(TEST_IMAGE_DIR, img_file)
    image_id = int(os.path.splitext(img_file)[0])  # 이미지 파일명의 숫자를 ID로 사용

    # 이미지 로드 및 변환
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 예측 수행
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # 결과 저장
    for i in range(len(predictions["boxes"])):
        bbox = predictions["boxes"][i].cpu().numpy()
        category_id = int(predictions["labels"][i].cpu().numpy())
        score = float(predictions["scores"][i].cpu().numpy())

        # 신뢰도 낮은 결과는 제외
        if score < 0.3:
            continue

        results.append([
            annotation_id, image_id, category_id,
            int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), score
        ])
        annotation_id += 1

# CSV 저장
df = pd.DataFrame(results, columns=["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"예측 결과가 {OUTPUT_CSV}에 저장되었습니다.")
