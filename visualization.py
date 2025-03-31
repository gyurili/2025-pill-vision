import os
import sys
import argparse

# 현재 파일의 경로를 기준으로 main 폴더 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# src 패키지에서 set_main_dir 함수 임포트 후 실행
from src import set_main_dir
set_main_dir()

# 모델 관련 함수 임포트
from models.evaluate import visualize_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="시각화할 이미지 인덱스를 입력하세요.")
    parser.add_argument("--idx", type=str, required=True, help="예측 결과를 시각화할 이미지 인덱스")
    args = parser.parse_args()

    # 명령줄 인자로 받은 값 사용
    visualize_bboxes(args.idx)