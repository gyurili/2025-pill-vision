import os
import sys

current_dir = os.getcwd()
project_dir = os.path.join(current_dir, '2025-health-vision')
sys.path.append(project_dir)

from models.RetinaNet.retinanet_func import *
from dataset.data_loader import get_dataloaders
from dataset.pill_dataset import TestDataset
def main():
    paths = setup_paths()
    train_loader, val_loader, test_loader = setup_dataloaders(paths, batch_size=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = 'C:/Users/user/OneDrive/Deesktop/codeit_Pr_1/final_model.pth'
    model = setup_model(device, checkpoint_path, num_classes=82)
    
    learning_rate = 0.0001
    weight_decay = 1e-5
    num_epochs = 16
    optimizer, lr_scheduler, scaler = setup_training_params(model, learning_rate, weight_decay, step_size=3, gamma=0.1)
    
    id_to_name, name_to_id = setup_category_mappings(paths)
    
    best_map = 0.0

    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, train_loader, device, scaler)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        val_map = evaluate_model(model, val_loader, device, epoch, iou_threshold=0.5)
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with mAP: {best_map:.4f}")
        
        lr_scheduler.step()
    
        model.load_state_dict(torch.load("best_model.pth"))
    test_model(model, test_loader, device)
    
if __name__ == '__main__':
    main()