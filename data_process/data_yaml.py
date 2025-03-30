import os, yaml

def write_data_yaml(data_path, drug_name):
  train_folder = "images/train"
  val_folder = "images/val"
	
  # Class names.
  class_names = list(drug_name.values())
  
  # Create data-set YAML configuration.
  data_config = {
    "path" : data_path,  # Root path of your data-set.
    "train" : train_folder,  # Training image folder.
    "val" : val_folder,  # Validation image folder.
    "nc" : len(class_names),  # Number of classes (82 classes).
    "names" : class_names  # List of class names.
	}

  # Save YAML configuration file.
  yaml_path = os.path.join(data_path, "custom_data.yaml")
  os.makedirs(os.path.dirname(yaml_path), exist_ok = True)
  with open(yaml_path, "w", encoding = "utf-8") as yaml_file:
    yaml.dump(data_config, yaml_file, default_flow_style = False, allow_unicode = True)

  print(f"✅ custom_data.yaml created successfully at {yaml_path}")