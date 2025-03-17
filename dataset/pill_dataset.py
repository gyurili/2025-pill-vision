import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def convert_bbox_format(bboxes, to_format="pascal"):
    """
    바운딩 박스 변환 함수.
    COCO <-> Pascal VOC 형식 변환 지원.

    Args:
        bboxes (list): 원본 바운딩 박스 리스트
        to_format (str): "pascal" 또는 "coco" 지정

    Returns:
        list: 변환된 바운딩 박스 리스트
    """
    converted_bboxes = []
    
    for bbox in bboxes:
        if to_format == "pascal":
            # COCO (x, y, w, h) → Pascal VOC (x_min, y_min, x_max, y_max)
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            converted_bboxes.append([x_min, y_min, x_max, y_max])
        
        elif to_format == "coco":
            # Pascal VOC (x_min, y_min, x_max, y_max) → COCO (x, y, w, h)
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            converted_bboxes.append([x_min, y_min, width, height])
        
        else:
            raise ValueError("Invalid format. Use 'pascal' or 'coco'.")

    return converted_bboxes


class PillDetectionDataset(Dataset):
    """
    객체 탐지 데이터셋 (Faster R-CNN, YOLO 등에서 사용 가능)
    """

    def __init__(self, df, image_dir, train=True, bbox_convert=False):
        """
        객체 탐지 데이터셋을 초기화합니다.

        Args:
            df (pd.DataFrame): 훈련 또는 검증 데이터셋
            image_dir (str): 이미지가 저장된 폴더 경로
            train (bool, optional): 학습 모드 여부 (True: 데이터 증강 포함). Defaults to True.
        """
        self.df = df
        self.image_dir = image_dir
        self.train = train
        self.bbox_convert = bbox_convert
        self.transforms = self.get_transforms()

    def __len__(self):
        """데이터셋의 길이를 반환합니다."""
        return len(self.df)

    def get_transforms(train=True):
        """
        Albumentations 변환 함수
        :param train: True일 경우 데이터 증강 적용, False면 검증용 변환만 적용
        :return: Albumentations 변환 객체
        """
        if train:
            return A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))  # bbox와 label을 함께 변환하도록 설정
        else:
            return A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))  # 검증용 변환도 동일하게 적용


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = eval(row["bbox"])
        boxes = convert_bbox_format(boxes)
        labels = eval(row["category_id"])

        # 데이터 증강 적용
        transformed = self.transforms(image=image, bboxes=boxes, category_id=labels)
        
        # 정규화 해제 (시각화를 위해)
        image_vis = transformed["image"].permute(1, 2, 0).cpu().numpy()  # (H, W, C) 형태
        image_vis = (image_vis * 0.5 + 0.5) * 255  # Albumentations Normalize 해제
        image_vis = image_vis.astype(np.uint8)

        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["category_id"], dtype=torch.int64)
        
        if self.bbox_convert == False:
            boxes = convert_bbox_format(boxes, "coco")
            boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target, image_vis


class TestDataset(Dataset):
    """
    주석이 없는 테스트 데이터셋을 로드하는 클래스
    """

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): 테스트 이미지가 저장된 폴더 경로
            transform (albumentations.Compose, optional): 이미지 변환을 위한 Albumentations 변환 객체
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform if transform else self.default_transforms()

    def __len__(self):
        """ 데이터셋 크기 반환 """
        return len(self.image_files)

    def __getitem__(self, idx):
        """ 이미지 로드 및 변환 """
        file_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, file_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 변환 적용
        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, file_name  # 라벨이 없으므로 파일명만 반환

    def default_transforms(self):
        """ 기본 이미지 변환 설정 """
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])