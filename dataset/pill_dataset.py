import os
import cv2
import torch
import numpy as np
import albumentations as A
import ast
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def convert_bbox_format(bboxes, to_format="pascal"):
    """
    바운딩 박스 변환 함수 (COCO ↔ Pascal VOC).

    Args:
        bboxes (list): 원본 바운딩 박스 리스트
        to_format (str): 변환할 형식 ("pascal" 또는 "coco")

    Returns:
        list: 변환된 바운딩 박스 리스트
    """
    converted_bboxes = []
    
    for bbox in bboxes:
        if to_format == "pascal":
            # COCO (x, y, w, h) → Pascal VOC (x_min, y_min, x_max, y_max)
            x_min, y_min, width, height = bbox
            converted_bboxes.append([x_min, y_min, x_min + width, y_min + height])
        
        elif to_format == "coco":
            # Pascal VOC (x_min, y_min, x_max, y_max) → COCO (x, y, w, h)
            x_min, y_min, x_max, y_max = bbox
            converted_bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
        
        else:
            raise ValueError("Invalid format. Use 'pascal' or 'coco'.")

    return converted_bboxes


class PillDetectionDataset(Dataset):
    """
    객체 탐지 데이터셋 (Faster R-CNN, YOLO 등에서 사용 가능).
    """

    def __init__(self, df, image_dir, train=True, bbox_convert=False):
        """
        데이터셋 초기화.

        Args:
            df (pd.DataFrame): 데이터셋 (CSV 파일 기반)
            image_dir (str): 이미지가 저장된 폴더 경로
            train (bool): 학습 모드 여부 (True: 데이터 증강 포함)
            bbox_convert (bool): False일 경우 bbox를 COCO 형식으로 변환
        """
        self.df = df
        self.image_dir = image_dir
        self.train = train
        self.bbox_convert = bbox_convert
        self.transforms = self.get_transforms()

    def __len__(self):
        """ 데이터셋 크기 반환 """
        return len(self.df)

    def get_transforms(self):
        """
        Albumentations 데이터 변환 설정.

        Returns:
            A.Compose: 변환된 이미지 및 바운딩 박스
        """
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5) if self.train else A.NoOp(),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["category_id"], min_visibility=0.05
        ))

    def __getitem__(self, idx):
        """
        개별 샘플을 반환.

        Args:
            idx (int): 데이터 인덱스

        Returns:
            tuple: (image, target, image_vis)
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 변환 (문자열 → 리스트)
        boxes = ast.literal_eval(row["bbox"])
        labels = ast.literal_eval(row["category_id"])
        boxes = convert_bbox_format(boxes, "pascal")  # COCO → Pascal 변환
        
        

        # 데이터 변환 적용
        transformed = self.transforms(image=image, bboxes=boxes, category_id=labels)

        # 시각화를 위한 정규화 해제
        image_vis = transformed["image"].permute(1, 2, 0).cpu().numpy()  # (H, W, C) 변환
        image_vis = ((image_vis * 0.5) + 0.5) * 255
        image_vis = image_vis.astype(np.uint8)

        # 최종 변환 데이터
        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["category_id"], dtype=torch.int64)

        # bbox_convert=False일 경우 다시 COCO 형식으로 변환 후 반환
        if not self.bbox_convert:
            boxes = torch.tensor(convert_bbox_format(boxes.tolist(), "coco"), dtype=torch.float32)

        target = {"boxes": boxes, "labels": labels}
        return image, target, image_vis


class TestDataset(Dataset):
    """
    주석이 없는 테스트 데이터셋.
    """

    def __init__(self, image_dir, transform=None):
        """
        테스트 데이터셋 초기화.

        Args:
            image_dir (str): 테스트 이미지 폴더 경로
            transform (albumentations.Compose, optional): 이미지 변환 설정
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform or self.default_transforms()

    def __len__(self):
        """ 데이터셋 크기 반환 """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        개별 테스트 샘플 반환.

        Args:
            idx (int): 데이터 인덱스

        Returns:
            tuple: (image, file_name)
        """
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
        """
        기본 이미지 변환 설정.

        Returns:
            A.Compose: Albumentations 변환 객체
        """
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])