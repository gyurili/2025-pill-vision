import os
import sys
import csv
import shutil
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# 현재 파일의 경로를 기준으로 main 폴더를 찾도록 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # main 폴더 경로
sys.path.append(BASE_DIR)  # main 폴더를 sys.path에 추가
        
# src 폴더의 config.py 가져오기
from src import set_main_dir
set_main_dir()

# data 폴더 내부의 모듈 import
from data_processing import *
from make_labels import new_label
from move_files import *
from data_yaml import write_data_yaml

def main():
    # annotation 경로 가져오기
    annotation_path = os.path.join(os.getcwd(), "image_annotations.csv")
    train_images_source = os.path.join(os.getcwd(), "train_images")  # 원본 학습 이미지 경로

    files_written = new_label(annotation_path)
    print(files_written)

    df = pd.read_csv(annotation_path)

    # 알약 이름 불러오기.
    drug_name = get_drug_name(df)
    num_unique_drug_names = count_unique_drug_names(annotation_path)
    print(f"Number of unique drug names: {num_unique_drug_names}.")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=24)

    # 파일 생성 및 이동.
    move_files(train_df, val_df)

    # 파일 제대로 이동시켰는지 확인
    train_image_dir = os.path.join(os.getcwd(), "yolov12/images/train")
    train_label_dir = os.path.join(os.getcwd(), "yolov12/labels/train")
    val_image_dir = os.path.join(os.getcwd(), "yolov12/images/val")
    val_label_dir = os.path.join(os.getcwd(), "yolov12/labels/val")

    train_result, train_message = check_files(train_image_dir, train_label_dir)
    val_result, val_message = check_files(val_image_dir, val_label_dir)

    print("Training Set: \t", train_message)
    print("Validation Set: ", val_message)

    # 훈련 및 검증 데이터 이미지 개수 확인
    train_count = count_files(train_image_dir)
    val_count = count_files(val_image_dir)

    print(f"Number of files in 'yolov12/images/train': {train_count}.")
    print(f"Number of files in 'yolov12/images/val': {val_count}.")

    # custom_data.yaml 구성하기.
    data_path = os.path.join(os.getcwd(), "yolov12")
    write_data_yaml(data_path, drug_name)

if __name__ == "__main__":
    set_main_dir()
    main()
