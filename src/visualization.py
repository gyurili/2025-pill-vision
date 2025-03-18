import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_sample(image, prediction):
    """
    Faster R-CNN의 예측 결과를 시각화하는 함수.

    Args:
        image (torch.Tensor): 모델 입력 이미지
        prediction (dict): Faster R-CNN 예측 결과
    """
    image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
    image = ((image * 0.5) + 0.5) * 255  # 정규화 해제
    image = image.astype(np.uint8)

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, f"Class {label}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
