import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import convert_bbox_format
from src.config import device

# ===========================
# 한글 폰트 설정 (운영체제별 선택)
# ===========================

# Windows
plt.rc("font", family="Malgun Gothic")

# Mac
# plt.rc("font", family="AppleGothic")

# Linux (Ubuntu 등)
# plt.rc("font", family="NanumGothic")  # 또는 "DejaVu Sans", "Noto Sans CJK KR"

# 마이너스(-) 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


def visualize_sample(image_vis, target, bbox_convert=True, class_names=None):
    """
    정답 바운딩 박스 시각화 (Pascal VOC).
    """
    boxes = target["boxes"].cpu().numpy().astype(int)
    labels = target["labels"].cpu().numpy()

    if bbox_convert:
        boxes = convert_bbox_format(boxes, to_format="pascal")

    plt.figure(figsize=(8, 8))
    plt.imshow(image_vis)

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        class_text = class_names.get(label, f"ID {label}") if class_names else f"ID {label}"

        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="green", facecolor="none"
        ))

        plt.text(x_min, y_min - 5, class_text, fontsize=10, color="white",
                 bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))

    plt.axis("off")
    plt.show()


def random_color():
    return np.random.rand(3,)  # RGB 0~1 for matplotlib


def draw_bboxes_plt(image, boxes, labels, scores, class_names=None, threshold=0.5):
    """
    matplotlib 기반 바운딩 박스 + 라벨 + 점수 시각화 (예측 결과).
    """
    h, w, _ = image.shape
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        color = random_color()
        class_name = class_names.get(label, f"Class {label}") if class_names else f"Class {label}"
        text = f"{class_name}: {score:.2f}"

        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor=color, facecolor="none"
        ))

        plt.text(x_min, y_min - 5, text, fontsize=10, color="white",
                 bbox=dict(facecolor=color, alpha=0.7, edgecolor="none"))

    plt.axis("off")
    plt.show()


def predict_and_visualize_dataset(model, test_dataset, class_names=None, threshold=0.5, num_samples=5):
    """
    모델로 테스트 이미지 예측 후 시각화 (matplotlib + 한글 대응).
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i in range(num_samples):
            image, file_name = test_dataset[i]
            image_tensor = image.unsqueeze(0).to(device)

            outputs = model(image_tensor)

            # 이미지 복원
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = ((image_np * 0.5) + 0.5) * 255
            image_np = image_np.astype(np.uint8)

            # 예측 결과 추출
            pred_boxes = outputs["pred_boxes"][0].cpu().numpy()
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_scores = pred_logits.softmax(-1).max(-1)[0].cpu().numpy()
            pred_labels = pred_logits.argmax(-1).cpu().numpy()

            # matplotlib 기반 시각화
            draw_bboxes_plt(
                image_np, pred_boxes, pred_labels, pred_scores,
                class_names=class_names, threshold=threshold
            )