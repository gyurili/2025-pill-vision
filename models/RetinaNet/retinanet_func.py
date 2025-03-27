import json
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision.ops.boxes import box_iou
import csv
import os
import sys
import torch
import torch.optim as optim
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torch.utils.data import DataLoader
import csv
from dataset.data_loader import get_dataloaders
from dataset.pill_dataset import TestDataset



# json 어노테이션 파일 읽어오기 (카테고리 매핑)
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 학습 함수
def train(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0
    for images, targets, _ in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # taget이 boxes와 labels를 가지지 않으면 콤마 뒤 에러 문장 return
        for target in targets:
            assert 'boxes' in target and 'labels' in target, "targets는 'boxes'와 'labels'를 포함해야 합니다."
        
        optimizer.zero_grad()
        
        # 메모리 절약을 위해 amp 사용
        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses.item()
    return total_loss / len(data_loader)

# 단순 AP 계산 함수
def calc_AP(pred_scores, matched_mask):
    # confidence score 기준 정렬
    sorted_indices = np.argsort(-pred_scores)
    matched_mask = matched_mask[sorted_indices]

    # True Positive, False Positive 누적합 계산
    tp_cumsum = np.cumsum(matched_mask)
    fp_cumsum = np.cumsum(~matched_mask)

    # recall & precision 계산
    recalls = tp_cumsum / (tp_cumsum[-1] + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)


    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    # Monotonicity 적용하여 면적을 더 정확하게 적분함
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    return np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])

# mAP 계산 함수
def calculate_metrics(targets, outputs, iou_threshold=0.5):
    # 카테고리 별 값 저장을 위한 딕셔너리
    class_aps = {i: [] for i in range(1, 83)}

    # 실제 값과 비교
    for target, output in zip(targets, outputs):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        pred_boxes = output["boxes"]
        pred_scores = output["scores"]
        pred_labels = output["labels"]

        # 예측값 없으면 AP값 0으로 치환 -> 에러 방지용
        if len(pred_boxes) == 0:
            for label in gt_labels.unique():
                class_aps[label.item()].append(0)
            continue

        # IOU 행렬 계산 함수
        ious = box_iou(gt_boxes, pred_boxes)

        # 클래스 별 AP 계산
        for cls in range(1, 83):
            # 현재 클래스와 동일한지 확인을 위한 변수 정의
            gt_cls_mask = gt_labels == cls
            pred_cls_mask = pred_labels == cls

            # 현재 클래스와 동일한지 판단
            gt_cls_boxes = gt_boxes[gt_cls_mask]
            pred_cls_boxes = pred_boxes[pred_cls_mask]
            pred_cls_scores = pred_scores[pred_cls_mask]

            # 예측값 없으면 0으로 치환 -> 에러 방지용
            if len(pred_cls_boxes) == 0 and len(gt_cls_boxes) > 0:
                class_aps[cls].append(0)
                continue
            if len(gt_cls_boxes) == 0:
                continue

            # 해당 클래스 IOU 계산
            ious_cls = box_iou(gt_cls_boxes, pred_cls_boxes)
            matches = []
            gt_matched = [False] * len(gt_cls_boxes)
            pred_matched = [False] * len(pred_cls_boxes)

            # IOU 값 기준으로 실제 값과 매칭
            for i in range(len(gt_cls_boxes)):
                for j in range(len(pred_cls_boxes)):
                    if (ious_cls[i, j] > iou_threshold and 
                        not gt_matched[i] and not pred_matched[j]):
                        matches.append((i, j))
                        gt_matched[i] = True
                        pred_matched[j] = True
            
            # 매칭 여부 판단
            matched_mask = torch.zeros(len(pred_cls_boxes), dtype=torch.bool)
            for _, pred_idx in matches:
                matched_mask[pred_idx] = True

            # 위에서 정의한 calc_AP 함수 활용해서 AP 계산
            ap = calc_AP(pred_cls_scores.cpu().numpy(), matched_mask.cpu().numpy())
            class_aps[cls].append(ap)

    # AP 평균 즉, mAP 계산
    mean_aps = []
    for cls in range(1, 83):
        if class_aps[cls]:
            mean_aps.append(np.mean(class_aps[cls]))
    return np.mean(mean_aps) if mean_aps else 0.0

# validation용 함수
def evaluate_model(model, data_loader, device, epoch, iou_threshold=0.5):
    model.eval()
    all_aps = []
    with torch.no_grad():
        for images, targets, _ in tqdm(data_loader):
            images = [img.to(device) for img in images]
            # target에 대한 딕셔너리 리스트 생성
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 메모리 절약을 위한 amp 사용
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)

            # 위에서 정의한 함수 기반 mAP값 계산
            ap = calculate_metrics(targets, outputs, iou_threshold)
            all_aps.append(ap)
    
    # 모든 배치에 대한 mAP 값 계산
    mean_ap = np.mean(all_aps)
    print(f"Epoch {epoch} - mAP@{iou_threshold}: {mean_ap:.4f}")
    return mean_ap

# test를 위한 함수
def test_model(model, data_loader, device):
    model.eval()
    final_predictions = []
    # 어노테이션을 1부터 저장
    annotation_id = 1
    # 최고 성능을 보인 파라미터 지정
    conf_thresh = 0.5
    nms_thresh = 0.3

    # 메모리 및 학습 속도 최적화를 위해 기울기 연산 X
    with torch.no_grad():
        for images, filenames in tqdm(data_loader):
            images = [img.to(device) for img in images]

            # 메모리 절약을 위한 amp 사용
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)

            for output, filename in zip(outputs, filenames):

                # confidence score 기반 필터링
                keep_idx = torch.where(output['scores'] >= conf_thresh)[0]
                if keep_idx.numel() == 0:
                    continue
                
                # 예측값 저장
                boxes = output['boxes'][keep_idx]
                scores = output['scores'][keep_idx]
                labels = output['labels'][keep_idx]

                # NMS 적용
                keep_nms = torch.ops.torchvision.nms(boxes, scores, nms_thresh)
                if keep_nms.numel() == 0:
                    continue
                
                # 임계값과 NMS에서 살아남은 값 저장
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]

                # 이미지 아이디 저장
                try:
                    image_id = int(os.path.splitext(filename)[0])
                except ValueError:
                    print(f"Warning: Could not convert {filename} to integer ID")
                    image_id = filename

                # 바운딩 박스 좌표 저장
                for i in range(len(boxes)):
                    xmin, ymin, xmax, ymax = boxes[i].tolist()  # 원본 이미지 : 976 * 1280 / 학습 이미지 : 640 * 640
                    bbox_x = int(xmin * 1.525)  # 원본 이미지 크기에 맞게 좌표 수정
                    bbox_y = int(ymin * 2)  # 원본 이미지 크기에 맞게 좌표 수정
                    bbox_w = int((xmax - xmin) * 1.525)  # 원본 이미지 크기에 맞게 좌표 수정
                    bbox_h = int((ymax - ymin) *2)  # 원본 이미지 크기에 맞게 좌표 수정
                    category_id = labels[i].item()
                    score = scores[i].item()

                    # csv 파일에 저장을 위한 전처리
                    final_predictions.append([
                        annotation_id, image_id, category_id,
                        bbox_x, bbox_y, bbox_w, bbox_h, score
                    ])
                    annotation_id += 1

    #csv 파일에 결과값 저장
    with open("predictions.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
        writer.writerows(final_predictions)
    print("All predictions saved to predictions.csv")


# 프로젝트 내 경로 설정 함수
def setup_paths():
    current_dir = os.getcwd()
    project_dir = os.path.join(current_dir, '2025-health-vision')
    sys.path.append(project_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    csv_path = os.path.join(data_dir, 'image_annotations.csv')
    image_dir = os.path.join(data_dir, "train_images")
    category_mapping_path = os.path.join(data_dir, "category_mapping.json")
    category_name_mapping_path = os.path.join(data_dir, "category_name_mapping.json")
    test_dir = os.path.join(data_dir, "test_images")
    
    return {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "csv_path": csv_path,
        "image_dir": image_dir,
        "category_mapping_path": category_mapping_path,
        "category_name_mapping_path": category_name_mapping_path,
        "test_dir": test_dir
    }

# 데이터 로더를 생성 함수
def setup_dataloaders(paths, batch_size=8):
    """학습/검증 및 테스트 데이터 로더를 생성하는 함수."""
    train_loader, val_loader = get_dataloaders(
        csv_path=paths["csv_path"],
        image_dir=paths["image_dir"],
        bbox_convert=True,
        batch_size=batch_size
    )
    
    test_dataset = TestDataset(paths["test_dir"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# 모델 생성 및 체크포인트 로드 함수
def setup_model(device, checkpoint_path, num_classes=82):
    torch.cuda.empty_cache()
    model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    return model

# 학습을 위한 파라미터 지정 함수
def setup_training_params(model, learning_rate=0.0001, weight_decay=1e-5, step_size=3, gamma=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler = torch.amp.GradScaler()
    return optimizer, lr_scheduler, scaler

# 카테고리 이름 및 매핑 정보 로드 함수
def setup_category_mappings(paths):
    category_name_mapping = load_json(paths["category_name_mapping_path"])
    category_mapping = load_json(paths["category_mapping_path"])
    id_to_name = {int(k): v for k, v in category_name_mapping.items()}
    name_to_id = {int(v): k for k, v in category_mapping.items()}
    return id_to_name, name_to_id