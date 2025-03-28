# 📌 Pill Detection with Object Detection Models

## 📖 프로젝트 개요

이 프로젝트는 다양한 배경과 각도에서 촬영된 정상 알약 이미지와 COCO 형식의 정보(이미지 및 바운딩 박스)를 통해, 다음의 **총 6가지 Object Detection 모델**을 활용하여 개체 탐지 및 클래스 예측 모델을 구현하는 프로젝트입니다.

각 모델의 성능을 실험적으로 비교한 후, 가장 성능이 우수했던 **YOLOv12n** 모델을 `main` 브랜치에 적용하였습니다. 나머지 모델들의 코드는 각각 별도 브랜치에 구현되어 있습니다.

### 사용한 모델 목록:

- YOLOv12n (`main` branch)
- Faster R-CNN
- RetinaNet
- Cascade R-CNN
- Deformable DETR
- EfficientNet

---

## 📂 폴더 구성 및 main 브랜치 구조 설명

```
2025-HEALTH-VISION/
├── .github/                 # GitHub 관련 설정 파일
├── data/                   # 데이터 디렉터리 (GitHub에는 포함되어 있지 않음)
│   ├── test_images/        # 테스트 이미지 (로컬에서 직접 다운로드 필요)
│   ├── train_images/       # 학습 이미지 (로컬에서 직접 다운로드 필요)
│   ├── train_annotations/  # 학습 어노테이션 (로컬에서 직접 다운로드 필요)
│   ├── category_mapping.json
│   ├── category_name_mapping.json
│   ├── image_annotations.csv
│
├── dataset/
│   ├── data_loader.py
│   ├── pill_dataset.py
│
├── models/
│   ├── model1.py           # YOLOv12n 기반 모델
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── visualization.py
│
├── environment.yml
├── main.py
├── test.py
├── README.md
```

---

### 📁 main 브랜치 구성 설명

- `data/` 폴더: 실제 학습과 테스트에 사용되는 이미지 및 어노테이션 데이터가 포함됨 (직접 다운로드 필요)
- `data_process/` 폴더: YOLO 학습용 데이터 전처리 스크립트 모음
  - `data_processing.py`: 고유한 카테고리 값으로 데이터 필터링 수행
  - `data_yaml.py`: YOLO 학습용 `data.yaml` 경로 설정 파일 생성
  - `make_labels.py`: YOLO 형식의 annotation TXT 파일 생성
  - `move_files.py`: train/val 이미지 폴더 생성 및 분할
  - `main.py`: 위 전처리 코드들을 순차 실행하는 통합 스크립트
- `data.yaml`: YOLO 학습을 위한 경로 설정 파일 생성
- `make_labels.py`: YOLO 형식의 annotation TXT 파일 생성
- `move_files.py`: train/val 이미지 폴더 생성 및 분할
- `main.py`: 위 전처리 코드들을 순차 실행하는 통합 스크립트

📁 `models/` 폴더
- `train.py`: 학습 실행
- `evaluate.py`: 테스트 이미지 기반 성능 평가 (CSV 결과 생성)
- `tuning.py`: 하이퍼파라미터 튜닝 코드
- `model_main.py`: 모델 관련 코드 전체 실행 스크립트

📁 `src/` 폴더
- `config.py`: 현재 경로를 main 디렉터리 기준으로 고정


📝 주의 사항
- `model_main.py` 실행 시 **wandb 계정 연동 필요** (로그인 필요)

---

## 🔧 사용 방법

### 1️⃣ 라이브러리 설치

```bash
conda env create -f environment.yml
conda activate health-vision
```

### 2️⃣ 전처리 실행

데이터를 준비한 후, 아래 명령어를 통해 YOLO 학습을 위한 전처리를 자동으로 실행합니다:

```bash
python data_process/main.py   # data_process 폴더 내 스크립트들 자동 실행
```

### 3️⃣ 모델 학습 및 평가

```bash
python models/model_main.py   # train.py, evaluate.py 등 실행
```

※ 이때 wandb 계정 연동이 필요할 수 있으므로 계정 정보를 미리 준비해주세요.


---

## 📈 예시 이미지 정보

- 파일: `K-003544-012247-016548-021026_0_2_0_2_70_000_200.png`
- 라벨: `무코스타정(레바미피드)(비매품)`
- 바운딩 박스: `[623, 870, 217, 213]`

---

## 🧼 데이터 전처리 안내

AI Hub에서 데이터를 직접 다운로드한 경우 다음과 같은 전처리 과정을 거쳐야 합니다:

1. **이미지 통합 정리**

   - 다운로드된 원본 이미지들은 여러 개의 하위 폴더에 나뉘어 저장되어 있습니다.
   - `notebooks/data_preprocessing.ipynb` 파일에 포함된 코드를 실행하여 **하나의 폴더로 통합**합니다.

2. **JSON 어노테이션 수정**

   - 원본 JSON 파일은 형식 오류가 있거나 모델 학습에 맞지 않게 구성되어 있을 수 있습니다.
   - 노트북 파일 내 수정 코드를 실행해 **올바른 포맷으로 정제**합니다.

3. **손상된 이미지 제거**

   - 일부 이미지 파일은 깨져 있거나 열 수 없는 경우가 있습니다.
   - 이를 자동으로 감지하여 제거하는 코드가 노트북에 포함되어 있습니다.

4. **CSV 변환**

   - 수정된 JSON 어노테이션을 COCO 형식 대신 **CSV 포맷**으로 변환하는 코드도 함께 제공됩니다.

모든 코드는 `notebooks/data_preprocessing.ipynb`에서 순차적으로 실행할 수 있습니다.

---

## 🔗 데이터 출처

- 본 프로젝트는 AI Hub에서 제공하는 공개 데이터셋을 활용하였습니다.
- 데이터셋 링크: [AI Hub - 의약품 이미지 객체 검출](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115\&topMenu=100\&dataSetSn=576)