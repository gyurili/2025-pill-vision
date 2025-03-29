import pandas as pd

# Reads the image_annotation.csv file to extract the file names,
# bounding boxes, image weight and height, and classes
# to create a new label that matches the format required by
# YOLOv8/YOLOv12 → class x_center y_center width height → e.g. 2 0.64 0.312 0.12 0.43.
# Where, x_center is the bounding box's x-coordinate normalized by being divided by the image's width,
# y_center is the bounding box's y-coordinate nromalized by being divided by the image's height,
# width is the bounding box's width that is normalized by being divided by the image's width,
# and finally height is the bounding box's height that is normalized by being divided by the image's height.
# So for an image with four pills, the corresponding .txt file
# will contain four lines for each of the pills in the format as required above.
import os
import pandas as pd

def new_label(csv_path):
    df = pd.read_csv(csv_path)
    files_written = 0  # 생성된 파일 개수
    
    for index, row in df.iterrows():
        category_id = eval(row.category_id)
        bbox = eval(row.bbox)  # [x_min, y_min, width, height]
        width, height = row.width, row.height  # 이미지 크기
        file_name = row.file_name.split(".png")[0]

        # 파일 저장 경로 설정
        file_path = os.path.join(os.path.dirname(csv_path), "txt_data")  # CSV 파일이 있는 디렉토리

        os.makedirs(file_path, exist_ok=True)  # 폴더가 없으면 생성

        # Windows 경로 문제 해결
        txt_file_path = os.path.join(file_path, f"{file_name}.txt")

        # 디버깅용 출력
        print(f"Saving file: {txt_file_path}")

        # UTF-8 인코딩으로 YOLO 형식의 라벨 파일 생성
        with open(txt_file_path, "w", encoding="utf-8") as file:
            for i in range(len(bbox)):
                normalized_bbox = [
                    (bbox[i][0] + bbox[i][2] / 2) / width,  # x_center
                    (bbox[i][1] + bbox[i][3] / 2) / height,  # y_center
                    bbox[i][2] / width,  # bbox width
                    bbox[i][3] / height  # bbox height
                ]
                line = f"{category_id[i]} " + " ".join(map(str, normalized_bbox))
                file.write(line + "\n")

            files_written += 1

    return files_written
