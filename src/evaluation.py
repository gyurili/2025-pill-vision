from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
from src import CLASS_NAMES, device

def convert_to_coco_format(dataset):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    ann_id = 1
    for idx in range(len(dataset)):
        image, target, _ = dataset[idx]
        image_id = idx

        coco["images"].append({
            "id": image_id,
            "height": image.shape[1],
            "width": image.shape[2]
        })

        for box, label in zip(target["boxes"], target["labels"]):
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1

    for class_id, class_name in CLASS_NAMES.items():
        coco["categories"].append({
            "id": class_id,
            "name": class_name
        })

    return coco

def get_predictions_in_coco_format(model, dataset, score_threshold=0.05):
    model.eval()
    predictions = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            image, _, _ = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)

            boxes = output["pred_boxes"][0].cpu().numpy()
            scores = output["pred_logits"][0].softmax(-1).cpu().numpy()
            labels = scores.argmax(-1)
            confs = scores.max(-1)

            for box, score, label in zip(boxes, confs, labels):
                if score < score_threshold or label == len(CLASS_NAMES):  # background 제외
                    continue
                x_min, y_min, x_max, y_max = box * 640  # 정규화 해제
                width = x_max - x_min
                height = y_max - y_min

                predictions.append({
                    "image_id": idx,
                    "category_id": int(label),
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "score": float(score)
                })

    return predictions


def evaluate_map(model, val_loader, class_names=None):
    val_dataset = val_loader.dataset

    print("Converting ground truth to COCO format...")
    coco_gt_dict = convert_to_coco_format(val_dataset)
    with open("gt.json", "w") as f:
        json.dump(coco_gt_dict, f)

    print("Generating predictions...")
    preds = get_predictions_in_coco_format(model, val_dataset)
    with open("preds.json", "w") as f:
        json.dump(preds, f)

    coco_gt = COCO("gt.json")
    coco_dt = coco_gt.loadRes("preds.json")

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = [0.5]  # mAP@0.5만 계산
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n클래스별 mAP@50:")
    precisions = coco_eval.eval['precision']  # shape: [iou, recall, cls, area, maxDet]
    iou_idx = 0  # IoU=0.5
    area_idx = 0  # area='all'
    max_det_idx = 2  # maxDet=100

    for cls_idx, cat_id in enumerate(coco_eval.params.catIds):
        # [recall,] → 평균값 (NaN 제외)
        precision = precisions[iou_idx, :, cls_idx, area_idx, max_det_idx]
        mean_prec = np.nanmean(precision)

        class_name = class_names[cat_id] if class_names and cat_id < len(class_names) else f"Class {cat_id}"
        print(f"{class_name:20s} | mAP@50: {mean_prec:.3f}")


def generate_submission_csv(model, test_dataset, output_path, threshold=0.5, orig_w=976, orig_h=1280):
    model.eval()
    model.to(device)
    
    annotation_id = 0
    results = []

    with torch.no_grad():
        for img_tensor, file_name in tqdm(test_dataset, desc="Generating Predictions"):
            image = img_tensor.unsqueeze(0).to(device)
            outputs = model(image)

            pred_boxes = outputs["pred_boxes"][0].cpu()  # (num_queries, 4), normalized
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_scores = pred_logits.softmax(-1).max(-1)[0].numpy()
            pred_labels = pred_logits.argmax(-1).numpy()

            # 정규화 해제: 예측 박스 (x_min, y_min, x_max, y_max)
            pred_boxes[:, [0, 2]] *= orig_w
            pred_boxes[:, [1, 3]] *= orig_h

            image_id = int(os.path.splitext(file_name)[0])

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score < threshold or label >= 82:
                    continue

                x_min, y_min, x_max, y_max = box.tolist()
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min

                results.append([
                    annotation_id,
                    image_id,
                    label,
                    round(x_min, 2),
                    round(y_min, 2),
                    round(bbox_w, 2),
                    round(bbox_h, 2),
                    round(float(score), 4)
                ])
                annotation_id += 1

    # CSV 저장
    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
        writer.writerows(results)

    print(f"CSV 저장 완료: {output_path}")