import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

def convert_bbox_format(bboxes, to_format="pascal", skip_conversion=False):
    """
    바운딩 박스 변환 함수 (COCO ↔ Pascal VOC).
    - `skip_conversion=True`이면 변환을 건너뜀 (즉, 원본 데이터 그대로 유지).
    """
    if skip_conversion:
        return bboxes  # 변환하지 않고 그대로 반환

    converted_bboxes = []
    
    for bbox in bboxes:
        if to_format == "pascal":
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            converted_bboxes.append([x_min, y_min, x_max, y_max])
        elif to_format == "coco":
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            converted_bboxes.append([x_min, y_min, width, height])
        else:
            raise ValueError("Invalid format. Use 'pascal' or 'coco'.")

    return converted_bboxes

class PillDetectionDataset(Dataset):
    """
    객체 탐지 데이터셋 (Faster R-CNN, YOLO 등에서 사용 가능).
    """
    def __init__(self, df, image_dir, train=True, use_conversion=False):
        self.df = df
        self.image_dir = image_dir
        self.train = train
        self.use_conversion = use_conversion  # 변환 여부 옵션
        self.transforms = self.get_transforms()
    
    def __len__(self):
        return len(self.df)
    
    def get_transforms(self):
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5) if self.train else A.NoOp(),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 가져오기
        boxes = eval(row["bbox"])  # 문자열로 저장된 리스트 변환
        labels = eval(row["category_id"])  

        # 변환 옵션에 따라 적용
        boxes = convert_bbox_format(boxes, "pascal", skip_conversion=not self.use_conversion)

        # 바운딩 박스 정렬 보장 (x_min <= x_max, y_min <= y_max)
        boxes = np.array(boxes, dtype=np.float32)
        boxes[:, 0], boxes[:, 2] = np.minimum(boxes[:, 0], boxes[:, 2]), np.maximum(boxes[:, 0], boxes[:, 2])
        boxes[:, 1], boxes[:, 3] = np.minimum(boxes[:, 1], boxes[:, 3]), np.maximum(boxes[:, 1], boxes[:, 3])

        h, w = image.shape[:2]

        # 정규화 수행
        boxes[:, [0, 2]] /= w  # x_min, x_max 정규화
        boxes[:, [1, 3]] /= h  # y_min, y_max 정규화

        # 바운딩 박스가 너무 작은 경우 필터링
        valid_boxes = []
        valid_labels = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            if x_max > x_min and y_max > y_min:  # 유효한 박스만 저장
                valid_boxes.append([x_min, y_min, x_max, y_max])
                valid_labels.append(labels[i])

        if len(valid_boxes) == 0:
            print(f"[WARN] Index {idx}: 모든 바운딩 박스가 무효합니다. 기본값으로 설정.")

        transformed = self.transforms(image=image, bboxes=valid_boxes, category_id=valid_labels)
        
        # 바운딩 박스 복원
        bboxes = np.array(transformed["bboxes"])
        bboxes[:, [0, 2]] *= w  # x_min, x_max 원본 크기 복원
        bboxes[:, [1, 3]] *= h  # y_min, y_max 원본 크기 복원

        image_vis = transformed["image"].permute(1, 2, 0).cpu().numpy()
        image_vis = ((image_vis * 0.5) + 0.5) * 255
        image_vis = image_vis.astype(np.uint8)

        image = transformed["image"]
        boxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(valid_labels, dtype=torch.int64)

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
            image_dir (str): 테스트 이미지가 저장된 폴더 경로
            transform (albumentations.Compose, optional): 이미지 변환 설정
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform if transform else self.default_transforms()

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