import matplotlib.pyplot as plt
import cv2
import numpy as np
from dataset import convert_bbox_format

# 한글 폰트 설정 (Windows)
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


def visualize_sample(image_vis, target, bbox_convert=True, class_names=None):
    """
    바운딩 박스를 시각화하는 함수.

    Args:
        image_vis (numpy.ndarray): 정규화 해제된 NumPy 이미지.
        target (dict): 바운딩 박스 정보 (COCO 또는 Pascal VOC 형식).
        bbox_convert (bool, optional): COCO 형식을 Pascal VOC로 변환할지 여부. Defaults to True.
        class_names (dict, optional): 클래스 ID와 이름 매핑 딕셔너리. Defaults to None.

    Returns:
        None
    """
    # 바운딩 박스 및 라벨 변환
    boxes = target["boxes"].cpu().numpy().astype(int)
    labels = target["labels"].cpu().numpy()

    if bbox_convert:
        boxes = convert_bbox_format(boxes, to_format="pascal")

    # 이미지 출력 준비
    plt.figure(figsize=(8, 8))
    plt.imshow(image_vis)
    
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box

        # 바운딩 박스 그리기 (초록색)
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="green", facecolor="none"
        ))

        # 클래스 이름 표시 (한글 지원)
        class_text = class_names.get(label, f"ID {label}") if class_names else f"ID {label}"
        plt.text(x_min, y_min - 5, class_text, fontsize=10, color="white",
                 bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))

    plt.axis("off")
    plt.show()