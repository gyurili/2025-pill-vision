import argparse
import optuna
import torch
import os
import sys

# 프로젝트 루트 디렉터리 설정 (main 폴더 기준)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)  # main 폴더를 sys.path에 추가

# src에서 set_main_dir() 가져오기
from src import set_main_dir  
set_main_dir()  # main 폴더를 작업 디렉토리로 설정

# 필요한 모듈 import
from tuning import objective
from train import train
from evaluate import evaluate
from src.config import TEST_IMAGE_PATH

test_image_path = TEST_IMAGE_PATH

def main():
    parser = argparse.ArgumentParser(description="YOLO Hyperparameter Optimization")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")  # 기본값 20
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # study.optimize()에 objective 함수 전달 시 num_epochs, device를 함께 전달
    study = optuna.create_study(direction="maximize")  # Maximize mAP@50.
    study.optimize(lambda trial: objective(trial, args.epochs, device), n_trials=10)

    print("Best augmentation and model's hyper-parameters: ", study.best_params)

    model = train(study.best_params, args.epochs, device)  # argparse에서 받은 epochs 사용
    evaluate(model, test_image_path)

if __name__ == "__main__":
    main()
