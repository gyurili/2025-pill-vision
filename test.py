from pathlib import Path
from dataset import get_dataloaders, TestDataset
from src import visualize_sample, CLASS_NAMES

if __name__ == "__main__":
    """스크립트 실행 시 데이터로더를 생성하고 검증"""
    
    # 현재 파일이 위치한 디렉토리를 기준으로 경로 설정
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / "./data/image_annotations.csv"
    IMAGE_DIR = BASE_DIR / "./data/train_images"
    TEST_DIR = BASE_DIR / "./data/test_images"

    # 훈련 & 검증 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(CSV_PATH, IMAGE_DIR, batch_size=8, val_split=0.2)
    test_dataset = TestDataset(TEST_DIR)

    # 데이터 확인
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    # 이미지와 타겟 데이터 분리
    image, target, image_vir = train_batch[0][0], train_batch[1][0], train_batch[2][0]

    # 시각화 실행
    visualize_sample(image_vir, target, True, CLASS_NAMES)