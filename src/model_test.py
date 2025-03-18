import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 랜덤 색상 생성 함수
def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

# 바운딩 박스를 이미지에 그리는 함수
def draw_bboxes(image, boxes, labels, scores, class_names=None, threshold=0.5):
    """
    이미지에 바운딩 박스를 그리는 함수.
    
    Args:
        image (numpy array): 원본 이미지
        boxes (numpy array): 예측된 바운딩 박스 (x_min, y_min, x_max, y_max)
        labels (numpy array): 클래스 라벨
        scores (numpy array): confidence score
        class_names (dict): 클래스 ID -> 클래스명 매핑 딕셔너리
        threshold (float): confidence score 임계값
    """
    image = image.copy()
    h, w, _ = image.shape

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue  # 스코어가 임계값보다 낮으면 무시

        color = random_color()  # 랜덤 색상
        x_min, y_min, x_max, y_max = box
        x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # 라벨 표시
        class_name = class_names.get(label, f"Class {label}") if class_names else f"Class {label}"
        text = f"{class_name}: {score:.2f}"
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# 테스트 데이터로 예측 수행 및 시각화
def predict_and_visualize_dataset(model, test_dataset, device, class_names=None, threshold=0.5, num_samples=5):
    """
    모델로 테스트 데이터셋을 예측하고 바운딩 박스를 시각화하는 함수.

    Args:
        model (torch.nn.Module): 학습된 모델
        test_dataset (Dataset): 테스트 데이터셋
        device (str): 실행할 디바이스 (cuda or cpu)
        class_names (dict): 클래스 ID -> 클래스명 매핑 딕셔너리
        threshold (float): confidence score 임계값
        num_samples (int): 시각화할 샘플 개수
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i in range(num_samples):
            image, file_name = test_dataset[i]  # Dataset에서 이미지 가져오기
            image_tensor = image.unsqueeze(0).to(device)  # 배치 차원 추가 및 GPU로 이동

            outputs = model(image_tensor)

            # 이미지 변환 (텐서 → numpy)
            image_np = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
            image_np = ((image_np * 0.5) + 0.5) * 255  # 정규화 해제
            image_np = image_np.astype(np.uint8)

            # 모델 예측 결과 가져오기
            pred_boxes = outputs["pred_boxes"][0].cpu().numpy()  # (x_min, y_min, x_max, y_max)
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_scores = pred_logits.softmax(-1).max(-1)[0].cpu().numpy()  # 최고 확률값
            pred_labels = pred_logits.argmax(-1).cpu().numpy()  # 가장 높은 확률의 클래스 ID

            # 시각화
            visualized_img = draw_bboxes(image_np, pred_boxes, pred_labels, pred_scores, class_names, threshold)

            # 결과 출력
            plt.figure(figsize=(8, 8))
            plt.imshow(visualized_img)
            plt.axis("off")
            plt.title(f"Predictions for {file_name}")
            plt.show()