import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

MAX_SIZE = 800

def convert_bbox_format(bboxes, to_format="pascal", skip_conversion=False):
    if skip_conversion:
        return bboxes  

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
        h, w, _ = image.shape
        scale = min(MAX_SIZE / w, MAX_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)  
        
        resized_image = cv2.resize(image, (new_w, new_h))

        padded_image = np.zeros((MAX_SIZE, MAX_SIZE, 3), dtype=np.uint8)
        
        pad_x = (MAX_SIZE - new_w) // 2
        pad_y = (MAX_SIZE - new_h) // 2
        padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image  

        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            x_min = x_min * scale + pad_x
            y_min = y_min * scale + pad_y
            x_max = x_max * scale + pad_x
            y_max = y_max * scale + pad_y

            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            new_bboxes.append([x_min, y_min, x_max, y_max])

        return padded_image, new_bboxes, pad_x, pad_y, scale
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = np.array(eval(row["bbox"]), dtype=np.float32)
        labels = eval(row["category_id"])  

        padded_image, new_bboxes, pad_x, pad_y, scale = self.pad_and_resize(image, boxes)

        normalized_bboxes = [[x / MAX_SIZE, y / MAX_SIZE, x2 / MAX_SIZE, y2 / MAX_SIZE] for x, y, x2, y2 in new_bboxes]

        transformed = self.transforms(image=padded_image, bboxes=normalized_bboxes, category_id=labels)

        restored_bboxes = np.array(transformed["bboxes"]) * MAX_SIZE

        image = transformed["image"]
        boxes = torch.tensor(restored_bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target, image


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform if transform else self.default_transforms()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
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
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
