import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class PillDetectionDataset(Dataset):
    def __init__(self, df, image_dir, train=True):
        """
        객체 탐지 데이터셋 (Faster R-CNN, YOLO 등에서 사용 가능)
        
        :param df: 데이터프레임 (훈련 또는 검증 데이터)
        :param image_dir: 이미지가 저장된 폴더 경로
        :param train: 학습 모드 여부 (True이면 데이터 증강 적용)
        """
        self.df = df
        self.image_dir = image_dir
        self.train = train
        self.transforms = self.get_transforms()

    def __len__(self):
        return len(self.df)

    def get_transforms(self):
        """
        트랜스폼을 정의하는 함수 (Albumentations 사용)
        """
        if self.train:
            return A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))
        else:
            return A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = eval(row["bbox"])  # 리스트 형태로 변환
        labels = eval(row["category_id"])

        transformed = self.transforms(image=image, bboxes=boxes, category_id=labels)
        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["category_id"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target