import logging
import os
from detectron2.utils.logger import setup_logger as setup_detectron_logger

def setup_logger(name, output_dir=None):
    """
    로거 설정 함수
    
    Args:
        name: 로거 이름
        output_dir: 로그 파일 저장 경로 (기본값: None)
        
    Returns:
        logger: 설정된 로거 객체
    """
    # Detectron2 로거 설정 사용
    logger = setup_detectron_logger(
        name=name,
        output_dir=output_dir,
        distributed_rank=0
    )
    
    # 로그 파일로 저장할 경우
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%m/%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger