import pandas as pd
from pill_dataset import PillDetectionDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def get_dataloaders(csv_path, image_dir, batch_size=8, val_split=0.2, shuffle=True):
    """
    훈련 데이터와 검증 데이터로 나누어 데이터로더 생성
    :param csv_path: CSV 파일 경로
    :param image_dir: 이미지 폴더 경로
    :param batch_size: 배치 크기
    :param val_split: 검증 데이터 비율
    :param shuffle: 데이터 섞기 여부
    :return: 훈련 데이터로더(train_loader), 검증 데이터로더(val_loader)
    """
    df = pd.read_csv(csv_path)

    # 데이터 분할 (train:val = (1 - val_split) : val_split)
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42, shuffle=True)

    # 데이터셋 생성
    train_dataset = PillDetectionDataset(train_df, image_dir, train=True)
    val_dataset = PillDetectionDataset(val_df, image_dir, train=False)

    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: tuple(zip(*batch))
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch))
    )

    return train_loader, val_loader