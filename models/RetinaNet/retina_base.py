import os
import sys

# 파일 경로 설정
current_dir = os.getcwd()
project_dir = os.path.join(current_dir, '2025-health-vision')
sys.path.append(project_dir)
data_dir = os.path.join(project_dir, 'data')

import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.ops.boxes import box_iou
from dataset.data_loader import get_dataloaders
from dataset.pill_dataset import TestDataset
from matplotlib import rcParams
import csv
from torch.utils.data import DataLoader
import torchsummary
rcParams['font.family'] = 'Malgun Gothic'

# CUDA 캐시 비우기
torch.cuda.empty_cache()

csv_path = os.path.join(data_dir, 'image_annotations.csv')
image_dir = os.path.join(data_dir, "train_images")
category_mapping_path = os.path.join(data_dir, "category_mapping.json")
category_name_mapping_path = os.path.join(data_dir, "category_name_mapping.json")
test_dir = os.path.join(data_dir, "test_images")

# 데이터 로더 불러오기
train_loader, val_loader = get_dataloaders(csv_path=csv_path, image_dir=image_dir, bbox_convert=True, batch_size=8)
test_dataset= TestDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 모델 불러오기 (RetinaNet)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = retinanet_resnet50_fpn_v2(weights="DEFAULT").to(device)
save_path = "final_model.pth"

# 하이퍼파라미터 설정
learning_rate = 0.0001
weight_decay = 1e-5
num_epochs = 12

# 옵티마이저 및 스케줄러 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scaler = torch.amp.GradScaler()

# JSON 파일 로드 함수
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

category_name_mapping = load_json(category_name_mapping_path)
category_mapping = load_json(category_mapping_path)
id_to_name = {int(k): v for k, v in category_name_mapping.items()}
name_to_id = {int(v): k for k, v in category_mapping.items()}

# 학습 함수
def train(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0
    for i, (images, targets, _) in enumerate(tqdm(data_loader)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses.item()
    return total_loss / (i + 1)

# AP 계산 함수
def calc_AP(pred_scores, matched_mask):
    sorted_indices = np.argsort(-pred_scores)  # 예측 점수 내림차순 정렬
    matched_mask = matched_mask[sorted_indices]

    tp_cumsum = np.cumsum(matched_mask)
    fp_cumsum = np.cumsum(~matched_mask)

    recalls = tp_cumsum / (tp_cumsum[-1] + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    return np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])

# 평가 지표 계산 함수 (mAP만 반환)
def calculate_metrics(targets, outputs, iou_threshold=0.3):
    all_aps = []

    for target, output in zip(targets, outputs):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        pred_boxes = output["boxes"]
        pred_scores = output["scores"]
        pred_labels = output["labels"]

        if len(pred_boxes) == 0:
            all_aps.append(0)
            continue

        ious = box_iou(gt_boxes, pred_boxes)  # IoU 계산

        # 매칭 추적
        matches = []
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                if ious[i, j] > iou_threshold and gt_labels[i] == pred_labels[j] and not gt_matched[i] and not pred_matched[j]:
                    matches.append((i, j))
                    gt_matched[i] = True
                    pred_matched[j] = True

        # 매칭된 예측값에 대한 마스크 생성
        matched_mask = torch.zeros(len(pred_boxes), dtype=torch.bool)
        for _, pred_idx in matches:
            matched_mask[pred_idx] = True

        # AP 계산
        ap = calc_AP(pred_scores.cpu().numpy(), matched_mask.cpu().numpy())
        all_aps.append(ap)

    mean_ap = np.mean(all_aps)

    return mean_ap

# 예측 결과 시각화 함수
def visualize_predictions(images, targets, outputs, epoch):
    num_images = min(5, len(images))
    for i in range(num_images):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).clip(0, 255).astype("uint8")  

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        # True 바운딩 박스 + 클래스 이름
        for box , label in zip(targets[i]["boxes"].cpu().numpy(), targets[i]["labels"].cpu().numpy()):
            class_name = id_to_name[label.item()]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(rect)
            ax.text(box[0], box[1], f"{class_name}", color="g", fontsize=10, weight="bold")

        # 예측된 바운딩 박스 + 클래스 이름
        for box, score, label in zip(outputs[i]["boxes"].cpu().numpy(), outputs[i]["scores"].cpu().numpy(), outputs[i]["labels"].cpu().numpy()):
            if score >= 0.5:
                class_name = id_to_name[label.item()]
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                ax.text(box[0], box[3] + 15, f"{class_name}: {score:.2f}", color="r", fontsize=10)

        plt.title(f'Epoch {epoch}')
        plt.show()

# 모델 검증증 함수 (mAP만 출력)
def evaluate_model(model, data_loader, device, epoch):
    model.eval()
    all_aps = []
    last_batch_images, last_batch_targets, last_batch_outputs = None, None, None
    image_ids = []

    with torch.no_grad():
        for images, targets, img_ids in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)

            image_ids.extend(img_ids)  # 이미지 ID 목록에 추가

            last_batch_images, last_batch_targets, last_batch_outputs = images, targets, outputs

            # mAP 계산
            ap = calculate_metrics(targets, outputs)
            all_aps.append(ap)
    mean_ap = np.mean(all_aps)

    print(f"Epoch {epoch} - mAP@0.5: {mean_ap:.4f}")


def test_model(model, data_loader, device):
    model.eval()
    final_predictions = []  # 최종 예측 결과를 저장할 리스트
    annotation_id = 1
    conf_thresh = 0.5
    nms_thresh = 0.3

    with torch.no_grad():
        for images, filenames in tqdm(data_loader):
            images = [img.to(device) for img in images]
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
            
            # outputs는 배치의 각 이미지에 대한 딕셔너리 리스트 (keys: 'boxes', 'scores', 'labels')
            for output, filename in zip(outputs, filenames):
                # 1. Confidence threshold 적용 (0.5 이상만 선택)
                keep_idx = torch.where(output['scores'] >= conf_thresh)[0]
                if keep_idx.numel() == 0:
                    continue

                boxes = output['boxes'][keep_idx]
                scores = output['scores'][keep_idx]
                labels = output['labels'][keep_idx]

                # 2. NMS 적용 (IoU threshold = 0.3)
                keep_nms = torch.ops.torchvision.nms(boxes, scores, nms_thresh)
                if keep_nms.numel() == 0:
                    continue

                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]

                # 3. 각 이미지에 대해, 파일명에서 image_id 추출 (예: "1.png" -> 1)
                try:
                    image_id = int(filename.split('.')[0])
                except:
                    image_id = filename

                # 4. 남은 각 탐지 결과에 대해 정보를 추출하여 final_predictions에 추가
                for i in range(len(boxes)):
                    xmin, ymin, xmax, ymax = boxes[i].tolist()
                    bbox_x = int(xmin)
                    bbox_y = int(ymin)
                    bbox_w = int(xmax - xmin)
                    bbox_h = int(ymax - ymin)
                    category_id = labels[i].item()
                    score = scores[i].item()

                    final_predictions.append([
                        annotation_id, image_id, category_id,
                        bbox_x, bbox_y, bbox_w, bbox_h, score
                    ])
                    annotation_id += 1

    # 최종 예측 결과를 CSV 파일에 저장
    with open("predictions.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
        writer.writerows(final_predictions)
    print("All predictions saved to predictions.csv")


# 학습 및 평가 루프
if __name__ == "__main__":
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, optimizer, train_loader, device, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        evaluate_model(model, val_loader, device, epoch)
        lr_scheduler.step()

    torch.save(model.state_dict(), save_path)
    print(f"Final model saved at epoch {epoch+1} to {save_path}")
    test_model(model, test_loader, device)
