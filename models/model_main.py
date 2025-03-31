import argparse
import optuna
import torch
import os
import sys
# 필요한 모듈 import
from models.tuning import objective
from models.train import train
from models.evaluate import evaluate
from src.config import TEST_IMAGE_PATH



def model_main():
    test_image_path = TEST_IMAGE_PATH
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
