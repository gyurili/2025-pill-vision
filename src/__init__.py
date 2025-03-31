import os
import sys

def set_main_dir():
    """어떤 환경에서 실행하든 'src' 폴더의 상위 디렉토리를 작업 디렉토리로 설정하는 함수"""
    current_file_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로
    src_dir = os.path.dirname(current_file_path)

    # 'src' 폴더의 상위 디렉토리로 설정
    parent_dir = os.path.dirname(src_dir)

    os.chdir(parent_dir)  # 'src' 폴더의 상위 디렉토리를 현재 작업 디렉토리로 설정
    return parent_dir  # 필요하면 상위 폴더 경로 반환

# 실행하면 현재 디렉토리가 'src'의 상위 폴더로 설정됨
if __name__ == "__main__":
    print("설정된 작업 디렉토리:", set_main_dir())
