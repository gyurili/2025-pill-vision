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

## 📂 폴더 구성
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
- 데이터셋 링크: [AI Hub - 의약품 이미지 객체 검출](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=576)

## 🚀 후속 개발 계획
- [x] 데이터셋 전처리 (이미지 크기 통일, 증강 등)
- [x] 모델 구현 및 학습 (YOLOv12n)
- [ ] 기타 모델 브랜치 병합 및 결과 정리
- [ ] 성능 평가 및 리포트 작성
- [ ] ONNX 또는 TensorRT로 배포