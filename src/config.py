import torch
import json
from pathlib import Path

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 프로젝트 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

# 카테고리 ID ↔ 약품명 매핑 파일 경로
category_mapping_path = BASE_DIR / "data/category_name_mapping.json"

# JSON 파일 로드 및 변환
with open(category_mapping_path, "r", encoding="utf-8") as f:
    category_name_mapping = json.load(f)

# class_name 딕셔너리 생성 (new_category_id → 약품명)
CLASS_NAMES = {int(k): v for k, v in category_name_mapping.items()}