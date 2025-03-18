import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .pill_dataset import PillDetectionDataset


def default_collate_fn(batch):
    """
    DataLoader에서 사용할 기본 collate 함수.

    Args:
        batch (list): 배치 데이터 리스트

    Returns:
        tuple: 언패킹된 배치 데이터
    """
    return tuple(zip(*batch))


def get_dataloaders(csv_path, image_dir, bbox_convert=False, batch_size=8, val_split=0.2):
    """
    훈련 및 검증 데이터로 분할 후 DataLoader를 생성.

    Args:
        csv_path (str or Path): 주석 파일 (CSV) 경로
        image_dir (str or Path): 이미지가 저장된 폴더 경로
        bbox_convert (bool, optional): False일 경우 COCO 형식 bbox 유지, True면 Pascal VOC 변환. Defaults to False.
        batch_size (int, optional): 배치 크기. Defaults to 8.
        val_split (float, optional): 검증 데이터 비율. Defaults to 0.2.

    Returns:
        tuple: 훈련 데이터로더 (`train_loader`), 검증 데이터로더 (`val_loader`)
    """
    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # 데이터 분할 (train : val = (1 - val_split) : val_split)
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42, shuffle=True)

    # 데이터셋 생성
    train_dataset = PillDetectionDataset(train_df, image_dir, train=True, bbox_convert=bbox_convert)
    val_dataset = PillDetectionDataset(val_df, image_dir, train=False, bbox_convert=bbox_convert)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_collate_fn)

    return train_loader, val_loader
