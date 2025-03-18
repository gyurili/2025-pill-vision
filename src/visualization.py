import matplotlib.pyplot as plt
import cv2
from dataset import convert_bbox_format


def visualize_sample(image, image_vis, target, class_id=False, bbox_convert=True):
    """
    바운딩 박스를 시각화하는 함수.

    Args:
        image (torch.Tensor): 모델 입력용 텐서.
        image_vis (numpy.ndarray): 정규화 해제된 NumPy 이미지.
        target (dict): 바운딩 박스 정보 (COCO 또는 Pascal VOC 형식).
        class_id (bool, optional): 클래스 ID를 출력할지 여부. Defaults to False.
        bbox_convert (bool, optional): COCO 형식을 Pascal VOC로 변환할지 여부. Defaults to True.
    
    Returns:
        None
    """
    # 바운딩 박스 및 라벨 변환
    boxes = target["boxes"].cpu().numpy().astype(int)
    labels = target["labels"].cpu().numpy()

    if bbox_convert:
        boxes = convert_bbox_format(boxes, to_format="pascal")

    # 바운딩 박스 시각화
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if class_id:
            class_text = f"ID {label}"
            cv2.putText(image_vis, class_text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # 이미지 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(image_vis)
    plt.axis("off")
    plt.show()