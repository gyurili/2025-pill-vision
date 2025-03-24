import torch
import os
from models.faster_rcnn import get_faster_rcnn_model
from dataset.data_loader import get_dataloaders
from torchmetrics.detection import MeanAveragePrecision
# pip install torchmetrics
from torchvision.ops import box_iou

CHECKPOINT_DIR = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트/Adam_0.0001/"
CSV_PATH = "/content/2025-health-vision/data/image_annotations.csv"
IMAGE_DIR = "/content/drive/MyDrive/코드잇 초급 프로젝트/정리된 데이터셋/train_images"

# 최신 체크포인트 찾기
def find_latest_checkpoint():
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("faster_rcnn_epoch") and f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError("체크포인트가 없습니다.")
    checkpoint_files.sort(key=lambda x: int(x.split("epoch")[1].split(".pth")[0]), reverse=True)
    return os.path.join(CHECKPOINT_DIR, checkpoint_files[0])

# Precision & Recall 계산
def compute_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0, 0

    ious = box_iou(pred_boxes, gt_boxes)
    matches = (ious > iou_threshold).float()

    true_positives = matches.sum(dim=1) > 0
    false_positives = ~true_positives
    false_negatives = matches.sum(dim=0) == 0

    TP = true_positives.sum().item()
    FP = false_positives.sum().item()
    FN = false_negatives.sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    return precision, recall

if __name__ == "__main__":
    MODEL_PATH = find_latest_checkpoint()
    print(f"가장 최신 모델 로드: {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 82
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("모델이 로드되었습니다.")

    # Validation 데이터 로드
    _, val_loader = get_dataloaders(CSV_PATH, IMAGE_DIR, batch_size=8, val_split=0.2)

    # 성능 평가 객체 (mAP@0.5)
    metric = MeanAveragePrecision(iou_thresholds=[0.5])

    all_precision, all_recall = [], []

    for images, targets, _ in val_loader:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            predictions = model(images)

        preds = []
        gts = []
        for pred, target in zip(predictions, targets):
            keep = pred["scores"] > 0.3
            pred_boxes = pred["boxes"][keep].cpu()
            pred_labels = pred["labels"][keep].cpu()

            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            preds.append({"boxes": pred_boxes, "scores": pred["scores"][keep].cpu(), "labels": pred_labels})
            gts.append({"boxes": gt_boxes, "labels": gt_labels})

            # Precision & Recall 계산
            precision, recall = compute_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels)
            all_precision.append(precision)
            all_recall.append(recall)

        # mAP@0.5 계산
        metric.update(preds, gts)

    # 최종 성능 계산
    results = metric.compute()

    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)

    print("\n===== 모델 성능 평가 결과 =====")
    print(f"mAP@0.5: {results['map_50']:.4f}")  
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
