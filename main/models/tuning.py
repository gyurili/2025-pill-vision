import os, yaml, optuna, wandb
from ultralytics import YOLO
from src.config import HYP_PATH,DATA_YAML

hyp_path = HYP_PATH
yaml_path = DATA_YAML

# Use Optuna to fine-tune our YOLOv12n model and built-in augmentation (hyp.yaml).
def objective(trial, num_epochs, device):
    # Suggest new values for augmentations to be fine-tuned.
    hsv_h = trial.suggest_float("hsv_h", 0.005, 0.05)
    hsv_s = trial.suggest_float("hsv_s", 0.1, 0.9)
    hsv_v = trial.suggest_float("hsv_v", 0.1, 0.9)
    degrees = trial.suggest_float("degrees", 0.0, 30.0)
    translate = trial.suggest_float("translate", 0.0, 0.3)
    scale = trial.suggest_float("scale", 0.1, 1.0)
    shear = trial.suggest_float("shear", 0.0, 10.0)
    flipud = trial.suggest_float("flipud", 0.0, 1.0)
    fliplr = trial.suggest_float("fliplr", 0.0, 1.0)
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 1.0)

    # Model hyper-parameters to be fine-tuned.
    lr0 = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    lrf = trial.suggest_float("lrf", 0.01, 0.1)
    batch = trial.suggest_categorical("batch", [5, 10])
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    rect = trial.suggest_categorical("rect", [True, False])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    cos_lr = trial.suggest_categorical("cos_lr", [True, False])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.8, 0.9)
    warmup_bias_lr = trial.suggest_float("warmup_bias_lr", 0.1, 0.9)
    cls = trial.suggest_float("cls", 0.5, 2.0)
    box = trial.suggest_float("box", 2.0, 10.0)

    # Update hyp.yaml file.
    hyp_dict = {
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "flipud": flipud,
        "fliplr": fliplr,
        "mosaic": mosaic,
        "mixup": mixup,
        "lr0": lr0,
        "lrf": lrf,
        "batch": batch,
        "momentum": momentum,
        "rect": rect,
        "dropout": dropout,
        "cos_lr": cos_lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
        "cls": cls,
        "box": box
    }

    # Make hyp.yaml and write.
    os.makedirs(hyp_path, exist_ok=True)
    with open("hyp.yaml", "w") as f:
        yaml.dump(hyp_dict, f)

    # Start W&B logging for this trial.
    wandb.init(project="yolo-optuna", name=f"trial-{trial.number}", reinit=True)

    # Train YOLOv12n model with new augmentation settings.
    model = YOLO("yolo12n.pt")

    results = model.train(
        data=yaml_path,
        epochs=num_epochs,  # 수정된 부분
        imgsz=1280,
        **hyp_dict,
        device=device,  # 수정된 부분
        project="yolo-optuna",
        name=f"trial-{trial.number}",
        verbose=False,
        val=True
    )

    # Extract mAP@50.
    mAP50 = results.results_dict["metrics/mAP50(B)"]

    # Log to W&B
    wandb.log({"mAP@50": mAP50})
    wandb.finish()  # Close W&B run.

    return mAP50  # Maximizing mAP@50.
