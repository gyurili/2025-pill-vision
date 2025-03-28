import os, shutil
import pandas as pd



# 훈련 및 검증 데이터 세트로 분리하는데 둘 다 모든 클래스 지닌 샘플 균형 있게
# 섞어야 한다. 그래서, 분리 후, 훈련 및 검증 데이터 세트가 모든 82개에 
# 클래스 가졌는지 확인한다.
train_images_source = os.path.join(os.getcwd(), "data", 'train_images')



def check_unique(df):
  all_classes = list()
  for category_ids in df["category_id"]:
    classes = eval(category_ids)
    all_classes.extend(classes) # Flatten into a single list of class. # 여기 다시 확인 필요 
  unique_classes = list(set(all_classes))
  return unique_classes #  len(unique_classes)해서 82가 안 나오면 문제.

def move_files(train_df, val_df):
  train_unique_classes = check_unique(train_df)
  val_unique_classes = check_unique(val_df) 
  print(f"Number of unique classes in train data-set: {len(train_unique_classes)}.\n" + 
        f"Number of unique classes in validation data-set: {len(val_unique_classes)}.")
  
  # 1. 이미지 파일 불러오기.
  train_files = [f for f in train_df["file_name"].tolist() if os.path.exists(os.path.join(train_images_source, f))]
  val_files = [f for f in val_df["file_name"].tolist() if os.path.exists(os.path.join(train_images_source, f))]
  
  # 2. Define path sources and destination folders.
  # make_labels.py에서 생성한 텍스트 파일 경로는 labels_source.
  labels_source = os.path.join(os.getcwd(), "txt_data") # Path to generated labels.
  train_folder = os.path.join(os.getcwd(), "yolov12/images/train") # Destination for train data.
  val_folder = os.path.join(os.getcwd(), "yolov12/images/val") # Destination for validation data.
  labels_train_folder = os.path.join(os.getcwd(), "yolov12/labels/train") # Destination for train labels.
  labels_val_folder = os.path.join(os.getcwd(), "yolov12/labels/val") # Destination for validation labels.
  
  # 3. Create train and validation folders.
  os.makedirs(train_folder, exist_ok = True)
  os.makedirs(val_folder, exist_ok = True)
  os.makedirs(labels_train_folder, exist_ok = True)
  os.makedirs(labels_val_folder, exist_ok = True)
  
  # 4. Move images and labels to respective folders.
  for file_name in train_files:
    # Move image.
    image_source_path = os.path.join(train_images_source, file_name)
    image_destination_path = os.path.join(train_folder, file_name)
    shutil.copy(image_source_path, image_destination_path)

    # Move label
    label_source_path = os.path.join(labels_source, file_name.split(".png")[0] + ".txt")
    label_destination_path = os.path.join(labels_train_folder, file_name.split(".png")[0] + ".txt")
    os.rename(label_source_path, label_destination_path)

  for file_name in val_files:
    # Move image.
    image_source_path = os.path.join(train_images_source, file_name)
    image_destination_path = os.path.join(val_folder, file_name)
    shutil.copy(image_source_path, image_destination_path)

    # Move label.
    label_source_path = os.path.join(labels_source, file_name.split(".png")[0] + ".txt")
    label_destination_path = os.path.join(labels_val_folder, file_name.split(".png")[0] + ".txt")
    os.rename(label_source_path, label_destination_path)

# 각 파일과 라벨 제대로 이동시켰는지 확인한다.
def check_files(image_dir, label_dir):
  """
  Compares the number and names of PNG files in the image directory
  with the text files in the label directory.

  Args:
    image_dir (str): Path to the image directory (e.g., 'images/train').
    label_dir (str): Path to the label directory (e.g., 'labels/train').

  Returns:
    tuple: A tuple containing:
      - bool: True if the files match, False otherwise.
      - str: A message indicating the result of the comparison.
  """
  image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
  label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

  # Check if the number of files matches.
  if len(image_files) != len(label_files):
    return False, f"Mismatch in the number of files: {len(image_files)} images vs. {len(label_files)} labels."

  # Check if the file names match (without extensions).
  image_names = [f.split(".png")[0] for f in image_files]
  label_names = [f.split(".txt")[0] for f in label_files]

  if set(image_names) != set(label_names):
    return False, "Mismatch in file names between images and labels."

  return True, "All files match!"

# 훈련 및 검증 파일 개수 센다.
def count_files(directory):
  """Counts the number of files in a directory.

  Args:
    directory: The path to the directory.

  Returns:
    The number of files in the directory.
  """
  return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])