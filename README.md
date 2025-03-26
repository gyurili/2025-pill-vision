# 📌 Pill Detection with YOLOv12n

## 📖 프로젝트 개요
이 프로젝트는 연회색 배경에서 다양한 각도에서 찾아내지는 정상 알약 이미지와 COCO 형식의 정보(이미지 및 바운딩 벅스)를 통해, YOLOv12n 기반의 개체 탐지 및 클래스 예측 모델을 구현하는 프로젝트입니다.

---

## 📂 포른 구성
```
2025-HEALTH-VISION/
│── .github/                 # GitHub 관련 설정 파일
│── data/
│   ├── test_images/        # 테스트 이미지
│   ├── train_images/       # 트래인 이미지
│   ├── train_annotations/  # 트래인 어노티션 (JSON)
│   ├── category_mapping.json
│   ├── category_name_mapping.json
│   ├── image_annotations.csv
│
│── dataset/
│   ├── data_loader.py      # 데이터로더 구현
│   ├── pill_dataset.py     # 알약 데이터셋 클래스
│
│── models/
│   ├── model1.py           # YOLOv12n 기반 모델 구현
│
│── notebooks/
│   ├── data_preprocessing.ipynb  # 전처리 열과
│
│── src/
│   ├── config.py
│   ├── train.py            # 학습 시스템
│   ├── visualization.py    # 결과 시각화
│
│── environment.yml         # Conda 환경 설정
│── main.py                 # 메인 시스템
│── test.py                 # 테스트 시스템
│── README.md
```

---

## 🔧 사용 방법

### 1️⃣ 환경 설정
```bash
conda env create -f environment.yml
conda activate health-vision
```

### 2️⃣ 실행 방식
```bash
git clone https://github.com/yourname/2025-health-vision.git
cd 2025-health-vision
python main.py
```

---

## 📈 예시 이미지 정보
- 파일: `K-003544-012247-016548-021026_0_2_0_2_70_000_200.png`
- 라벨: `무코스타정(레바미피드)(비매품)`
- 바운딩 벅스: `[623, 870, 217, 213]`

---

## 🚀 후지 개발 계획
- [x] 데이터셀 전처리 (이미지 크기 통일, 증가 등)
- [x] YOLOv12n 모델 구현 및 학습
- [ ] 성능 평가 및 리포트 작성
- [ ] ONNX 또는 TensorRT로 배포
