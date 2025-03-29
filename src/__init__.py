import os
import sys

def set_main_dir():
    """어떤 환경에서 실행하든 'main' 폴더를 작업 디렉토리로 설정하는 함수"""
    current_file_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로
    main_dir = os.path.dirname(current_file_path)

    while os.path.basename(main_dir) != "main":
        main_dir = os.path.dirname(main_dir)
        if main_dir == os.path.dirname(main_dir):  # 루트까지 갔는데 main을 못 찾으면 오류 발생
            raise FileNotFoundError("main 폴더를 찾을 수 없습니다.")

    os.chdir(main_dir)  # 'main' 폴더를 현재 작업 디렉토리로 설정
    return main_dir  # 필요하면 main 폴더 경로 반환

# 실행하면 현재 디렉토리가 'main'으로 설정됨
if __name__ == "__main__":
    print("설정된 작업 디렉토리:", set_main_dir())