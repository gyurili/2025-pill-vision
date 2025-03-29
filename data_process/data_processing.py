import os, csv, shutil, yaml
import pandas as pd

def get_drug_name(df):
  # Create an empty dictionary to store the drug_name → categorical_id mapping.
  drug_name_to_categorical_id = dict()


  # Iterate through the rows of the data-frame.
  for _, row in df.iterrows():
    drug_names = eval(row["drug_name"])
    category_ids = eval(row["category_id"])
  
    # Map each category_id. to its corresponding drug_name.
    for drug_name, category_id in zip(drug_names, category_ids):
      drug_name_to_categorical_id[category_id] = drug_name

  # Now 'drug_name_to_categorical_id' holds the mapping.
  sorted_drug_name_to_categorical_id = {k : drug_name_to_categorical_id[k] for k in sorted(drug_name_to_categorical_id)}
  check_duplicate_names(sorted_drug_name_to_categorical_id) # 알약 이름 중복 확인.
  return sorted_drug_name_to_categorical_id

def check_duplicate_names(sorted_drug_name_to_categorical_id):
  # Check for duplicate drug names.
  duplicate_drug_names = {drug_name for drug_name, cat_id in sorted_drug_name_to_categorical_id.items()
                        if list(sorted_drug_name_to_categorical_id.values()).count(cat_id) > 1}

  if duplicate_drug_names:
    print(f"Duplicate drug names: {duplicate_drug_names}")
  else:
    print("No duplicate drug names found.")
  
def count_unique_drug_names(csv_path):
  """
  Counts the number of unique drug names in the 'drug_name' column of the CSV file.
  """
  df = pd.read_csv(csv_path)
  unique_drug_names = set()
  for drug_name_list in df['drug_name']:
    for drug_name in eval(drug_name_list):  # Evaluate the string as a list
      unique_drug_names.add(drug_name)

  return len(unique_drug_names)