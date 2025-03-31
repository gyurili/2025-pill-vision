import os
import sys

# 현재 파일의 경로를 기준으로 main 폴더 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# main 폴더를 sys.path에 추가하여 모듈을 찾을 수 있도록 설정
sys.path.append(BASE_DIR)

# src 패키지에서 set_main_dir 함수 임포트 후 실행
from src import set_main_dir
set_main_dir()

# 데이터 처리 및 모델 관련 모듈 임포트
from data_process.data_main import data_main
from models.model_main import model_main


if __name__ == "__main__":
    # 데이터 처리 실행
    data_main()
    # 모델 학습 및 평가 실행
    model_main()
