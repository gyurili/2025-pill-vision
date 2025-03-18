import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def visualize_sample(image, prediction, file_name, save_dir="output_images"):
    image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
    image = ((image * 0.5) + 0.5) * 255  # 정규화 해제
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    if len(boxes) == 0:
        print(f"감지된 객체 없음: {file_name}")
        return

    for box, label, score in zip(boxes, labels, scores):
        # 바운딩 박스를 원본 크기로 변환 (0~1 → 픽셀 크기)
        x_min, y_min, x_max, y_max = map(int, box * np.array([640, 640, 640, 640]))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, f"Class {label} ({score:.2f})", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 다시 RGB로 변환 (matplotlib에서 사용하기 위함)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 저장할 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"prediction_{file_name}")
    
    # SSH 환경에서는 `plt.show()` 대신 `savefig()` 사용
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predictions for {file_name}")
    plt.savefig(save_path)  # 이미지 저장
    plt.close()  # Matplotlib 리소스 해제
    
    print(f"저장 완료: {save_path}")
