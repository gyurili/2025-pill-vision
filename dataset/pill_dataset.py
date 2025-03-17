import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def convert_bbox_format(bboxes):
    """
    COCO 형식 [x, y, width, height] → Pascal VOC 형식 [x_min, y_min, x_max, y_max] 변환

    Args:
        bboxes (list): COCO 형식의 바운딩 박스 리스트

    Returns:
        list: Pascal VOC 형식으로 변환된 바운딩 박스 리스트
    """
    new_bboxes = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox[:4]
        x_max = x_min + width
        y_max = y_min + height

        if y_max <= y_min:
            print(f"경고: 잘못된 bbox 수정됨: {bbox}")
            y_max = y_min + abs(height)

        new_bboxes.append([x_min, y_min, x_max, y_max])

    return new_bboxes


class PillDetectionDataset(Dataset):
    """
    객체 탐지 데이터셋 (Faster R-CNN, YOLO 등에서 사용 가능)
    """

    def __init__(self, df, image_dir, train=True):
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
        self.transforms = self.get_transforms()

    def __len__(self):
        """데이터셋의 길이를 반환합니다."""
        return len(self.df)

    def get_transforms(self):
        """
        Albumentations 기반의 이미지 변환을 정의합니다.

        Returns:
            A.Compose: 이미지 변환 파이프라인
        """
        if self.train:
            return A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))

    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 가져옵니다.

        Args:
            idx (int): 샘플 인덱스

        Returns:
            tuple: 변환된 이미지와 대상(target) 딕셔너리
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["file_name"])

        # 이미지 로드 및 오류 처리
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 변환 적용 (COCO → Pascal VOC 변환)
        boxes = eval(row["bbox"])
        boxes = convert_bbox_format(boxes)
        labels = eval(row["category_id"])

        # 이미지 변환 적용
        transformed = self.transforms(image=image, bboxes=boxes, category_id=labels)
        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["category_id"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target