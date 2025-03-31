import numpy as np
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualize_detection_korean(image, predictions, class_names, font_path):
    """
    한글 지원 객체 탐지 시각화 함수
    
    Args:
        image: 입력 이미지 (OpenCV BGR 형식)
        predictions: 모델 예측 결과
        class_names: 클래스 이름 목록
        font_path: 한글 폰트 경로
        
    Returns:
        result_img: 시각화된 결과 이미지 (RGB 형식)
    """
    # 원본 이미지 복사 (BGR 형식 유지)
    result_img = image.copy()

    # 이미지 크기
    img_height, img_width = result_img.shape[:2]

    # 예측 결과 가져오기
    instances = predictions["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(int)
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    # 점수 기준으로 결과 정렬
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    classes = classes[indices]
    scores = scores[indices]

    # 랜덤 색상 생성 (클래스별로 다른 색상)
    random.seed(42)  # 일관된 색상을 위한 시드
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]

    # BGR -> RGB 변환 (PIL용)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 한글 폰트 로드
    font_size = 25
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 영역 추적을 위한 리스트
    text_regions = []

    # 각 탐지에 대해 박스 그리기
    for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        if score < 0.5:  # 점수 기준 필터링
            continue
        # 박스 좌표
        x0, y0, x1, y1 = box

        # 색상 선택 (RGB 형식)
        color = (colors[cls_id][0], colors[cls_id][1], colors[cls_id][2])

        # 박스 그리기
        line_width = 3  # 두꺼운 라인
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        # 클래스명과 점수
        class_name = class_names[cls_id]
        text = f"id:{cls_id}, {class_name}: {score:.2f}"

        # 텍스트 크기 측정
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 기본 텍스트 위치 (박스 상단)
        text_x = x0
        text_y = y0 - text_height - 10

        # 위치 조정 (이미지 경계 밖으로 나가는 경우)
        if text_y < 0:
            text_y = y1 + 5  # 박스 하단에 배치

        # 텍스트 영역 계산
        text_box = (text_x, text_y, text_x + text_width, text_y + text_height)

        # 다른 텍스트와 겹치는지 확인 및 위치 조정
        overlap = True
        position_options = [
            (x0, y0 - text_height - 10),  # 상단
            (x0, y1 + 5),                 # 하단
            (x1 - text_width, y0 - text_height - 10),  # 우측 상단
            (x1 - text_width, y1 + 5),    # 우측 하단
            (x0, (y0 + y1) // 2),         # 좌측 중앙
            (x1 - text_width, (y0 + y1) // 2)  # 우측 중앙
        ]

        for pos_x, pos_y in position_options:
            # 위치 조정 (이미지 경계 밖으로 나가는 경우)
            if pos_x < 0: pos_x = 0
            if pos_y < 0: pos_y = 0
            if pos_x + text_width > img_width: pos_x = img_width - text_width
            if pos_y + text_height > img_height: pos_y = img_height - text_height

            candidate_box = (pos_x, pos_y, pos_x + text_width, pos_y + text_height)

            # 기존 텍스트와 겹치는지 확인
            overlap = False
            for existing_box in text_regions:
                if check_overlap(candidate_box, existing_box):
                    overlap = True
                    break

            if not overlap:
                text_x, text_y = pos_x, pos_y
                text_box = candidate_box
                break

        # 텍스트 영역 추가
        text_regions.append(text_box)

        # 텍스트 배경 그리기
        draw.rectangle([text_box[0], text_box[1], text_box[2], text_box[3]], fill=color)

        # 텍스트 그리기 (흰색)
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    # 결과 numpy array로 변환 (RGB 형식)
    result_img = np.array(pil_img)

    return result_img

def check_overlap(box1, box2):
    """
    두 박스가 겹치는지 확인
    
    Args:
        box1: 첫 번째 박스 좌표 (x1, y1, x2, y2)
        box2: 두 번째 박스 좌표 (x1, y1, x2, y2)
        
    Returns:
        overlap: 겹침 여부 (True/False)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 겹치지 않는 조건
    if x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max:
        return False
    return True

def display_image(image, title=None, figsize=(12, 8)):
    """
    이미지 시각화 유틸리티 함수
    
    Args:
        image: 시각화할 이미지
        title: 제목 (기본값: None)
        figsize: 그림 크기 (기본값: (12, 8))
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()