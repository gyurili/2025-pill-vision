import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pill_dataset import PillDetectionDataset


def get_dataloaders(csv_path, image_dir, batch_size=8, val_split=0.2, shuffle=True):
    """
    훈련 데이터와 검증 데이터로 나누어 데이터로더를 생성합니다.

    Args:
        csv_path (str or Path): 주석 파일 (CSV) 경로
        image_dir (str or Path): 이미지가 저장된 폴더 경로
        batch_size (int, optional): 배치 크기. Defaults to 8.
        val_split (float, optional): 검증 데이터 비율. Defaults to 0.2.
        shuffle (bool, optional): 훈련 데이터 섞기 여부. Defaults to True.

    Returns:
        tuple: 훈련 데이터로더 (`train_loader`), 검증 데이터로더 (`val_loader`)
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


if __name__ == "__main__":
    """스크립트 실행 시 데이터로더를 생성하고 검증"""
    
    # 현재 파일이 위치한 디렉토리를 기준으로 경로 설정
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / "../data/image_annotations.csv"
    IMAGE_DIR = BASE_DIR / "../data/train_images"

    # 훈련 & 검증 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(CSV_PATH, IMAGE_DIR, batch_size=8, val_split=0.2)

    # 데이터 확인
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print(f"훈련 배치 크기: {len(train_batch[0])}")
    print(f"검증 배치 크기: {len(val_batch[0])}")