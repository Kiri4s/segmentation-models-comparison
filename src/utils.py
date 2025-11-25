from typing import Dict, Tuple

import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.segmentation import MeanIoU


def train_and_validate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    device: str,
    model_name: str,
    checkpoint_dir: str = "./checkpoints",
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    history = {"train_loss": [], "val_loss": [], "val_meanIoU": [], "tl": [], "vl": []}
    best_val_loss = float("inf")
    IoU = MeanIoU(num_classes=7, input_format="index").to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / len(train_loader))

            history["train_loss"].append(running_loss / len(train_loader))
            history["tl"].append(loss.item())

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                predicted = (torch.argmax(outputs, dim=1)).long()

                # val_bar.set_postfix(
                #    val_loss=val_loss / len(val_loader)
                # )

                val_loss /= len(val_loader)
                iou = IoU(predicted, masks)
                history["vl"].append(loss.item())
                history["val_meanIoU"].append(iou.item())

        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {history['train_loss'][-1]:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Mean IoU: {iou.item():.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f"{checkpoint_dir}/{model_name}_best.pth")

        save_history(history, f"{checkpoint_dir}/{model_name}_training_history.json")

    return model, history


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_history(history: Dict[str, list], path: str) -> None:
    with open(path, "w") as f:
        json.dump(history, f)
