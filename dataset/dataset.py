import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

def register_datasets(image_data_root, train_json="train_coco.json", val_json="test_coco.json"):
    """
    학습 및 검증 데이터셋 등록
    
    Args:
        image_data_root: 이미지 데이터 경로
        train_json: 학습 데이터 JSON 파일 이름
        val_json: 검증 데이터 JSON 파일 이름
    """
    # Detectron2 로거 설정
    setup_logger()
    
    # Train 데이터셋 등록
    register_coco_instances(
        "my_dataset_train",
        {},
        train_json,
        os.path.join(image_data_root, "images")
    )
    
    # Validation 데이터셋 등록
    register_coco_instances(
        "my_dataset_val",
        {},
        val_json,
        os.path.join(image_data_root, "images")
    )
    
    print("데이터셋 등록 완료: my_dataset_train, my_dataset_val")
    
def get_dataset_dicts(dataset_name):
    """
    데이터셋 딕셔너리 반환
    
    Args:
        dataset_name: 데이터셋 이름
        
    Returns:
        dataset_dicts: 데이터셋 딕셔너리
    """
    return DatasetCatalog.get(dataset_name)

def setup_test_metadata(classes, name="test_metadata"):
    """
    테스트용 메타데이터 설정
    
    Args:
        classes: 클래스 목록
        name: 메타데이터 이름
    """
    MetadataCatalog.get(name).thing_classes = classes
    return name