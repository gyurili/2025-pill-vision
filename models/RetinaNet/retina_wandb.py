import os
import sys

# 경로 설정
current_dir = os.getcwd()
project_dir = os.path.join(current_dir, '2025-health-vision')
sys.path.append(project_dir)

import json
import torch
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.ops.boxes import box_iou
from dataset.data_loader import get_dataloaders
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'

# CUDA 캐시 비우기
torch.cuda.empty_cache()

# W&B 로그인
wandb.login()

# 데이터 경로 설정
csv_path = "C:/Users/user/OneDrive/Deesktop/codeit_Pr_1/2025-health-vision/data/image_annotations.csv"
image_dir = "C:/Users/user/OneDrive/Deesktop/codeit_Pr_1/2025-health-vision/data/train_images"
category_mapping_path = "C:/Users/user/OneDrive/Deesktop/codeit_Pr_1/2025-health-vision/data/category_mapping.json"
category_name_mapping_path = "C:/Users/user/OneDrive/Deesktop/codeit_Pr_1/2025-health-vision/data/category_name_mapping.json"

# 데이터 로더 불러오기
train_loader, val_loader = get_dataloaders(csv_path=csv_path, image_dir=image_dir, bbox_convert=True, batch_size=8)

# 모델 불러오기 (RetinaNet)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = retinanet_resnet50_fpn_v2(weights="DEFAULT").to(device)

# 하이퍼파라미터 설정
learning_rate = 0.0001
weight_decay = 1e-5
num_epochs = 5

# 옵티마이저 및 스케줄러 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scaler = torch.amp.GradScaler()

# JSON 파일 로드 함수
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

category_name_mapping = load_json(category_name_mapping_path)
id_to_name = {int(k): v for k, v in category_name_mapping.items()}

# 평가 지표 계산 함수
def calculate_metrics(targets, outputs, iou_threshold=0.5, score_threshold=0.5):
    all_precisions, all_recalls = [], []

    for target, output in zip(targets, outputs):
        gt_boxes = target["boxes"]
        pred_boxes = output["boxes"]
        pred_scores = output["scores"]

        if len(pred_boxes) == 0:
            all_precisions.append(0)
            all_recalls.append(0)
            continue

        # Score threshold 적용
        valid_indices = pred_scores > score_threshold
        pred_boxes = pred_boxes[valid_indices]

        ious = box_iou(gt_boxes, pred_boxes)
        max_ious, _ = ious.max(dim=1)
        tp = (max_ious > iou_threshold).sum().item()
        fn = len(gt_boxes) - tp
        fp = len(pred_boxes) - tp

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        all_precisions.append(precision)
        all_recalls.append(recall)

    return np.mean(all_precisions), np.mean(all_recalls)

# mAP@0.5 계산 함수
def compute_map_50(targets, outputs, iou_threshold=0.5):
    aps = []
    for target, output in zip(targets, outputs):
        gt_boxes = target["boxes"]
        pred_boxes = output["boxes"]
        pred_scores = output["scores"]

        if len(pred_boxes) == 0:
            aps.append(0)
            continue

        # Score 순으로 정렬
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]

        ious = box_iou(gt_boxes, pred_boxes)
        max_ious, _ = ious.max(dim=1)
        tp = (max_ious > iou_threshold).sum().item()
        precision = tp / len(pred_boxes)
        aps.append(precision)

    return np.mean(aps)

# 모델 평가 함수
def evaluate_model(model, data_loader, device, epoch, score_threshold, nms_threshold):
    model.eval()
    all_precisions, all_recalls, all_map50 = [], [], []

    with torch.no_grad():
        for images, targets, _ in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)

            # NMS 적용
            for output in outputs:
                keep = torch.ops.torchvision.nms(output['boxes'], output['scores'], nms_threshold)
                output['boxes'] = output['boxes'][keep]
                output['scores'] = output['scores'][keep]
                output['labels'] = output['labels'][keep]

            precision, recall = calculate_metrics(targets, outputs, score_threshold=score_threshold)
            map50 = compute_map_50(targets, outputs)

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_map50.append(map50)

    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_map50 = np.mean(all_map50)

    wandb.log({"epoch": epoch, "precision": mean_precision, "recall": mean_recall, "mAP@0.5": mean_map50})
    print(f"Epoch {epoch} - Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, mAP@0.5: {mean_map50:.4f}")

# 모델 학습 함수
def train():
    wandb.init(project="retinanet_nms_confidence")
    score_threshold = wandb.config.score_threshold
    nms_threshold = wandb.config.nms_threshold

    for epoch in range(num_epochs):
        model.train()
        for images, targets, _ in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        evaluate_model(model, val_loader, device, epoch, score_threshold, nms_threshold)
        lr_scheduler.step()

# W&B Sweep 설정 (mAP@0.5 최대화)
sweep_config = {
    "method": "grid",
    "metric": {"name": "mAP@0.5", "goal": "maximize"},
    "parameters": {
        "score_threshold": {"values": [0.3, 0.5, 0.7]},
        "nms_threshold": {"values": [0.3, 0.5, 0.7]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="retinanet_nms_confidence")
wandb.agent(sweep_id, train)
