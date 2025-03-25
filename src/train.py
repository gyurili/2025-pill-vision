import torch
import torch.optim as optim
import os
from models.faster_rcnn import get_faster_rcnn_model
from dataset.data_loader import get_dataloaders

# 체크포인트 경로
CHECKPOINT_DIR = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 최신 체크포인트 확인 (실험별 폴더에서 찾기)
def find_latest_checkpoint(optimizer_name, lr):
    """해당 optimizer와 learning rate 조합의 최신 체크포인트 찾기"""
    experiment_folder = os.path.join(CHECKPOINT_DIR, f"{optimizer_name}_{lr}")
    if not os.path.exists(experiment_folder):
        return None, 0  # 폴더가 없으면 처음부터 시작

    checkpoint_files = [f for f in os.listdir(experiment_folder) if f.startswith("faster_rcnn_epoch") and f.endswith(".pth")]
    if not checkpoint_files:
        return None, 0  # 체크포인트 파일이 없으면 처음부터 시작

    checkpoint_files.sort(key=lambda x: int(x.split("epoch")[1].split(".pth")[0]), reverse=True)
    latest_checkpoint = os.path.join(experiment_folder, checkpoint_files[0])
    latest_epoch = int(checkpoint_files[0].split("epoch")[1].split(".pth")[0])
    return latest_checkpoint, latest_epoch

# 모델 학습
def train(model, train_loader, val_loader, num_epochs, optimizer_name="Adam", lr=0.0001, device="cuda"):
    model.to(device)

    # 옵티마이저 설정
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("지원되지 않는 옵티마이저")

    # 실험별 체크포인트 폴더 생성
    experiment_folder = os.path.join(CHECKPOINT_DIR, f"{optimizer_name}_{lr}")
    os.makedirs(experiment_folder, exist_ok=True)

    latest_checkpoint, start_epoch = find_latest_checkpoint(optimizer_name, lr)
    
    if latest_checkpoint:
        print(f"[INFO] 최신 체크포인트 {latest_checkpoint}에서 학습을 이어서 시작합니다.")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    else:
        print("[INFO] 체크포인트 없음. 처음부터 학습 시작.")

    loss_history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        print(f"[INFO] Epoch {epoch+1}/{num_epochs} 시작")
        
        for step, (images, targets, _) in enumerate(train_loader):
            if step % 20 == 0:
                print(f"[INFO] Step {step+1}/{len(train_loader)} 진행 중...")  
            
            images = list(image.to(device) for image in images)
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 실험별 폴더에 체크포인트 저장
        checkpoint_path = os.path.join(experiment_folder, f"faster_rcnn_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] Model checkpoint saved at {checkpoint_path}")

    return loss_history
