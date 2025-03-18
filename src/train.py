import torch
import torch.optim as optim
from models.faster_rcnn import get_faster_rcnn_model
from dataset.data_loader import get_dataloaders
import os

def train(model, train_loader, val_loader, num_epochs, lr=0.0001, device="cuda"):
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
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
            loss = torch.mean(torch.stack([v for v in loss_dict.values()]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join("models", f"faster_rcnn_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")
