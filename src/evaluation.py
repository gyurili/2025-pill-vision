from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import os
from tqdm import tqdm
import torch
from src import CLASS_NAMES

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

def get_predictions_in_coco_format(model, dataset, device, score_threshold=0.05):
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

def evaluate_map(model, val_loader, device):
    val_dataset = val_loader.dataset  # 여기서 dataset 가져옴

    print("Converting ground truth to COCO format...")
    coco_gt_dict = convert_to_coco_format(val_dataset)
    with open("gt.json", "w") as f:
        json.dump(coco_gt_dict, f)

    print("Generating predictions...")
    preds = get_predictions_in_coco_format(model, val_dataset, device)
    with open("preds.json", "w") as f:
        json.dump(preds, f)

    coco_gt = COCO("gt.json")
    coco_dt = coco_gt.loadRes("preds.json")

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = [0.5]  # mAP@50만 계산
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

