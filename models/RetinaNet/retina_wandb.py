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
from models.RetinaNet.retinanet_func import train, load_json
rcParams['font.family'] = 'Malgun Gothic'

# CUDA 캐시 비우기
torch.cuda.empty_cache()

# W&B 로그인
wandb.login()

# 데이터 경로 설정
csv_path = os.path.join(project_dir, 'data', 'image_annotations.csv')
image_dir = os.path.join(project_dir, 'data', 'train_images')
category_mapping_path = os.path.join(project_dir, 'data', 'category_mapping.json')
category_name_mapping_path = os.path.join(project_dir, 'data', 'category_name_mapping.json')

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

category_name_mapping = load_json(category_name_mapping_path)
id_to_name = {int(k): v for k, v in category_name_mapping.items()}


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
