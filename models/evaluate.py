import os
import csv
import cv2
import matplotlib.pyplot as plt
from src.config import TEST_LABELS_PATH, TEST_CSV_PATH, TEST_IMAGE_PATH

def evaluate(model, test_images_path):
    """모델 평가 및 결과를 CSV로 저장하는 함수"""
    model(test_images_path, save=True, save_txt=True, save_conf=True, imgsz=1280)

    test_label_path = TEST_LABELS_PATH
    test_csv = TEST_CSV_PATH
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    mapping_dict = {
        "249": 0, "572": 1, "1865": 2, "1899": 3, "2482": 4, "3350": 5, "3482": 6, "3543": 7, "3742": 8, "3831": 9,
        "4377": 10, "4542": 11, "5001": 12, "5093": 13, "5885": 14, "6191": 15, "6562": 16, "10220": 17, "10223": 18,
        "12080": 19, "12246": 20, "12419": 21, "12777": 22, "13160": 23, "13394": 24, "13899": 25, "16231": 26,
        "16261": 27, "16547": 28, "16550": 29, "16687": 30, "18109": 31, "18146": 32, "18356": 33, "19231": 34,
        "19551": 35, "19606": 36, "19860": 37, "20013": 38, "20237": 39, "20876": 40, "21025": 41, "21324": 42,
        "21770": 43, "22073": 44, "22346": 45, "22361": 46, "22626": 47, "23202": 48, "23222": 49, "24849": 50,
        "25366": 51, "25437": 52, "25468": 53, "27652": 54, "27732": 55, "27776": 56, "27925": 57, "27992": 58,
        "28762": 59, "29344": 60, "29450": 61, "29666": 62, "29870": 63, "30307": 64, "31704": 65, "31862": 66,
        "31884": 67, "32309": 68, "33008": 69, "33207": 70, "33877": 71, "33879": 72, "34596": 73, "35205": 74,
        "36636": 75, "37776": 76, "38161": 77, "38953": 78, "41767": 79, "44198": 80, "44833": 81
    }

    img_width, img_height = 976, 1280

    with open(test_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
        annotation_id = 1

        for filename in os.listdir(test_label_path):
            if not filename.endswith(".txt"):
                continue

            image_id = filename[:-4]
            filepath = os.path.join(test_label_path, filename)

            if not os.path.exists(filepath):
                print(f"Warning: Label file {filepath} not found. Skipping...")
                continue

            with open(filepath, 'r') as txtfile:
                for line in txtfile:
                    values = line.strip().split()
                    if len(values) < 6:
                        print(f"Warning: Skipping malformed line in {filename}: {line}")
                        continue

                    category_id = int(values[0])
                    if category_id not in mapping_dict.values():
                        print(f"Warning: Category ID {category_id} not found in mapping. Skipping...")
                        continue

                    category_name = list(mapping_dict.keys())[list(mapping_dict.values()).index(category_id)]
                    bbox_center_x, bbox_center_y = float(values[1]), float(values[2])
                    bbox_width, bbox_height = float(values[3]), float(values[4])
                    score = round(float(values[5]), 2)

                    bbox_x = int((bbox_center_x - bbox_width / 2) * img_width)
                    bbox_y = int((bbox_center_y - bbox_height / 2) * img_height)
                    bbox_w = int(bbox_width * img_width)
                    bbox_h = int(bbox_height * img_height)

                    csv_writer.writerow([annotation_id, image_id, category_name, bbox_x, bbox_y, bbox_w, bbox_h, score])
                    annotation_id += 1


def visualize_bboxes(image_id):
    """Bounding box를 이미지 위에 시각화하는 함수"""
    image_path = os.path.join(TEST_IMAGE_PATH, f"{image_id}.png")
    csv_path = TEST_CSV_PATH

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image at {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(csv_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for row in csv_reader:
            if row[1] == image_id:
                category_name, bbox_x, bbox_y, bbox_w, bbox_h, score = row[2], int(row[3]), int(row[4]), int(row[5]), int(row[6]), float(row[7])
                cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255, 0, 0), 2)
                cv2.putText(img, f"{category_name} {score}", (bbox_x, bbox_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()