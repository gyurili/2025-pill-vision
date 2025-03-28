import yaml
import wandb
from ultralytics import YOLO
from src.config import DATA_YAML

yaml_path = DATA_YAML




def train(best_params, num_epochs, device):
    with open("hyp_best.yaml", "w") as f:
        yaml.dump(best_params, f)

    wandb.init(project="yolo-optuna", name="best-run")

    model = YOLO("yolo12n.pt")
    model.train(
        data=yaml_path,
        epochs=num_epochs,  # 수정된 부분
        imgsz=1280,
        **best_params,
        device=device,  # 수정된 부분
        project="yolo-optuna-final",
        name="best-run"
    )

    wandb.finish()  # Close W&B run.

    return model
