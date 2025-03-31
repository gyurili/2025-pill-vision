# config.py

import os

# Define paths dynamically using os.path.join
HYP_PATH = os.path.join(os.getcwd(), "hyp_path")
DATA_YAML = os.path.join(os.getcwd(), "yolov12", "custom_data.yaml")
TEST_IMAGE_PATH = os.path.join(os.getcwd(), "data", "test_images")
TEST_LABELS_PATH = os.path.join(os.getcwd(), "yolo-optuna-final", "best-run2", "labels")
TEST_CSV_PATH = os.path.join(os.getcwd(), "yolov12", "test_labels.csv")
MODEL_PATH =  os.path.join(os.getcwd(), "yolo-optuna-final", "best-run", "weights","best.pt")
