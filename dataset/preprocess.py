import pandas as pd
import ast
import os
import json

def preprocess_annotations(csv_path, output_csv_path, mapping_path):
    """
    기존 category_id를 0부터 시작하는 새로운 category_id로 매핑하여 CSV 파일을 저장하는 함수

    Args:
        csv_path (str): 원본 CSV 파일 경로
        output_csv_path (str): 변환된 CSV 파일 저장 경로
        mapping_path (str): category_id 매핑을 저장할 JSON 파일 경로
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # category_id가 NaN이거나 빈 리스트인 경우 제거
    df = df[df["category_id"].notna()]
    df["category_id"] = df["category_id"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df[df["category_id"].apply(lambda x: isinstance(x, list) and len(x) > 0)]  # 빈 리스트 제거

    # 기존 category_id 리스트를 flatten하여 고유한 클래스 추출
    unique_categories = sorted(set([c for sublist in df["category_id"] for c in sublist]))

    # category_id를 0부터 시작하는 인덱스로 매핑
    category_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_categories)}

    print("category_id 매핑 테이블:")
    print(category_mapping)

    # category_id 변환 시 KeyError 방지 처리
    def map_category_ids(category_list):
        return [category_mapping[i] for i in category_list if i in category_mapping]

    # 기존 category_id를 새로운 인덱스로 변환
    df["category_id"] = df["category_id"].apply(map_category_ids)

    # 변환된 category_id가 0~num_classes-1 범위인지 확인
    max_category_id = max(category_mapping.values())
    print(f"총 클래스 개수: {max_category_id + 1}")

    # 새로운 CSV 파일 저장
    df.to_csv(output_csv_path, index=False)
    print(f"새로운 CSV 파일이 저장되었습니다: {output_csv_path}")

    # category_mapping을 JSON으로 저장
    with open(mapping_path, "w") as f:
        json.dump(category_mapping, f)
    print(f"category_id 매핑이 저장되었습니다: {mapping_path}")

    return category_mapping

if __name__ == "__main__":
    BASE_PATH = "/content/2025-health-vision/data"
    CSV_PATH = os.path.join(BASE_PATH, "image_annotations.csv")
    OUTPUT_CSV_PATH = os.path.join(BASE_PATH, "image_annotations_fixed.csv")
    MAPPING_PATH = os.path.join(BASE_PATH, "category_mapping.json")  # JSON 파일로 저장

    # 전처리 실행
    preprocess_annotations(CSV_PATH, OUTPUT_CSV_PATH, MAPPING_PATH)
