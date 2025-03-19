import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# 이미지 크기를 통일할 최대 크기 설정 (예: 800x800)
MAX_SIZE = 800

def convert_bbox_format(bboxes, to_format="pascal", skip_conversion=False):
    """ 바운딩 박스 변환 함수 (COCO ↔ Pascal VOC) """
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
    def __init__(self, df, image_dir, train=True):
        self.df = df
        self.image_dir = image_dir
        self.train = train
        self.transforms = self.get_transforms()
    
    def __len__(self):
        return len(self.df)
    
    def get_transforms(self):
        return A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))

    def pad_and_resize(self, image, bboxes):
        """ 
        원본 비율을 유지하면서 검정색 패딩을 추가하여 `MAX_SIZE x MAX_SIZE`로 맞춤 
        """
        h, w, _ = image.shape
        scale = min(MAX_SIZE / w, MAX_SIZE / h)  # 축소/확대 비율
        new_w, new_h = int(w * scale), int(h * scale)  # 비율 유지한 새로운 크기
        
        # 이미지 크기 조정
        resized_image = cv2.resize(image, (new_w, new_h))

        # 검정색 배경의 빈 캔버스 생성
        padded_image = np.zeros((MAX_SIZE, MAX_SIZE, 3), dtype=np.uint8)
        
        # 중앙 정렬 (좌표 계산)
        pad_x = (MAX_SIZE - new_w) // 2
        pad_y = (MAX_SIZE - new_h) // 2
        padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image  # 중앙 배치

        # 바운딩 박스 크기 변환 및 패딩 좌표 적용
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = x_min * scale + pad_x
            y_min = y_min * scale + pad_y
            x_max = x_max * scale + pad_x
            y_max = y_max * scale + pad_y

            # 변환 후 `x_min > x_max`, `y_min > y_max` 방지
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            # 바운딩 박스가 유효한 경우만 추가
            if x_max > x_min and y_max > y_min:
                new_bboxes.append([x_min, y_min, x_max, y_max])
            else:
                print(f"[WARNING] 잘못된 bbox 제거: ({x_min}, {y_min}, {x_max}, {y_max})")

        return padded_image, new_bboxes, pad_x, pad_y, scale
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 가져오기
        boxes = np.array(eval(row["bbox"]), dtype=np.float32)
        labels = eval(row["category_id"])  

        # 원본 비율 유지 + 패딩 추가
        padded_image, new_bboxes, pad_x, pad_y, scale = self.pad_and_resize(image, boxes)

        # Albumentations 적용
        transformed = self.transforms(image=padded_image, bboxes=new_bboxes, category_id=labels)

        # 최종 변환된 bbox 좌표 확인
        bboxes = np.array(transformed["bboxes"])
        #print(f"[DEBUG] Index {idx} - 최종 변환된 bbox: {bboxes}")

        image = transformed["image"]
        boxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target, image


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

        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, file_name

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