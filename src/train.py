import torch
from src import device

def train_model(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs, max_norm=0.1):
    """ 
    Deformable DETR 학습 및 검증 함수 
    
    Args:
        model: 학습할 모델 (Deformable DETR)
        criterion: 손실 함수
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        optimizer: 최적화 알고리즘
        scheduler: 학습률 스케줄러 (옵션)
        num_epochs: 총 학습 epoch 수
        max_norm: Gradient Clipping을 위한 최대 노름 (default: 0.1)
    
    Returns:
        model: 학습이 완료된 모델
        train_loss_history: 각 epoch의 학습 손실 값 리스트
        val_loss_history: 각 epoch의 검증 손실 값 리스트
    """
    print("Training started...\n")
    
    train_loss_history = []  # 학습 손실 값 저장 리스트
    val_loss_history = []    # 검증 손실 값 저장 리스트

    for epoch in range(num_epochs):
        # Training Step
        model.train()
        criterion.train()
        
        total_train_loss = 0.0
        for images, targets, _ in train_loader:
            if isinstance(images, (list, tuple)):
                images = torch.stack(images)  # 변환 추가

            images = images.to(device).float()
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            total_train_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)  # 학습 손실 값 저장
        
        # Validation Step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                if isinstance(images, (list, tuple)):
                    images = torch.stack(images)  # 변환 추가

                images = images.to(device).float()
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict.values())

                total_val_loss += losses.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)  # 검증 손실 값 저장
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    print("\nTraining completed.")
    return model, train_loss_history, val_loss_history