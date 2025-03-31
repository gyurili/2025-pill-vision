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
    
    # Add two different arguments for training and tuning epochs
    parser.add_argument("--train_epochs", type=int, default=10, help="Number of training epochs")  # Training epochs
    parser.add_argument("--tune_epochs", type=int, default=1, help="Number of tuning epochs")  # Tuning epochs

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Study optimization, using the tuning epochs
    study = optuna.create_study(direction="maximize")  # Maximize mAP@50.
    study.optimize(lambda trial: objective(trial, args.tune_epochs, device), n_trials=1)

    print("Best augmentation and model's hyper-parameters: ", study.best_params)

    # Train the model using the best hyperparameters and training epochs
    model = train(study.best_params, args.train_epochs, device)  # Pass in training epochs
    evaluate(model,test_image_path)
