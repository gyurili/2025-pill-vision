import matplotlib.pyplot as plt
import cv2

def visualize_sample(image, image_vis, target, class_names=None):
    """
    바운딩 박스를 시각화하는 함수.
    Args:
        image (torch.Tensor): 모델 입력용 텐서
        image_vis (numpy.ndarray): 정규화 해제된 NumPy 이미지
        target (dict): 바운딩 박스 정보
        class_names (list, optional): 클래스 ID에 해당하는 이름 리스트
    """
    boxes = target["boxes"].cpu().numpy().astype(int)
    labels = target["labels"].cpu().numpy()

    # 바운딩 박스 그리기
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if class_names:
            class_name = class_names[label] if label < len(class_names) else f"ID {label}"
            cv2.putText(image_vis, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_vis)
    plt.axis("off")
    plt.show()