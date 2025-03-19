import torch
import torch.optim as optim
import os
from models.faster_rcnn import get_faster_rcnn_model
from dataset.data_loader import get_dataloaders

# 체크포인트 경로
CHECKPOINT_DIR = "/content/drive/MyDrive/코드잇/초급 프로젝트/체크포인트"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Google Drive에서 가장 최신 체크포인트 찾기
def find_latest_checkpoint():
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("faster_rcnn_epoch") and f.endswith(".pth")]
    if not checkpoint_files:
        return None, 0  # 체크포인트 없으면면 처음부터 학습

    checkpoint_files.sort(key=lambda x: int(x.split("epoch")[1].split(".pth")[0]), reverse=True)
    
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
    latest_epoch = int(checkpoint_files[0].split("epoch")[1].split(".pth")[0])
    
    return latest_checkpoint, latest_epoch

def train(model, train_loader, val_loader, num_epochs, lr=0.0001, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 가장 최신 체크포인트 확인
    latest_checkpoint, start_epoch = find_latest_checkpoint()
    
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} (Classifier: {loss_dict['loss_classifier'].item():.4f}, BoxReg: {loss_dict['loss_box_reg'].item():.4f}, Obj: {loss_dict['loss_objectness'].item():.4f}, RPN: {loss_dict['loss_rpn_box_reg'].item():.4f})")
        
        # 체크포인트 저장
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"faster_rcnn_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] Model checkpoint saved at {checkpoint_path}")

    return loss_history
