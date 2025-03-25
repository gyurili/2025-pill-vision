import torch
import torch.amp as amp
from tqdm import tqdm
from src import device

def train_step(model, criterion, train_loader, optimizer, scaler, max_norm=1.0):
    model.train()
    criterion.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, targets, _ in progress_bar:
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        images = images.to(device).float()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

        if torch.isnan(losses) or torch.isinf(losses):
            print("Warning: NaN or Inf detected in loss. Skipping this batch.")
            continue

        scaler.scale(losses).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())

    return total_train_loss / len(train_loader)

def val_step(model, criterion, val_loader):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for images, targets, _ in progress_bar:
            if isinstance(images, (list, tuple)):
                images = torch.stack(images)
            images = images.to(device).float()
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

            total_val_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

    return total_val_loss / len(val_loader)

def train_model(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs, max_norm=1.0):
    print("Training started...\n")
    scaler = amp.GradScaler()
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_train_loss = train_step(model, criterion, train_loader, optimizer, scaler, max_norm)
        avg_val_loss = val_step(model, criterion, val_loader)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), "model_6.pth")

    print("\nTraining completed.")
    return model, train_loss_history, val_loss_history
