from typing import Dict, Tuple

import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torchvision import transforms as T


def get_model(model_name: str, cfg):
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            activation=cfg.activation,
        )
    elif model_name == "pspnet":
        model = smp.PSPNet(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            activation=cfg.activation,
        )

    elif model_name == "deeplab":
        model = smp.DeepLabV3(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            activation=cfg.activation,
        )

    elif model_name == "fpn":
        cfg.transform = T.Compose([T.Resize((2464, 2464)), T.ToTensor()])
        cfg.target_transform = T.Compose([T.Resize((2464, 2464))])
        model = smp.FPN(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            activation=cfg.activation,
        )

    elif model_name == "unet++":
        cfg.transform = T.Compose([T.Resize((2464, 2464)), T.ToTensor()])
        cfg.target_transform = T.Compose([T.Resize((2464, 2464))])
        model = smp.UnetPlusPlus(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            activation=cfg.activation,
        )

    else:
        raise ValueError(
            f"Model {model_name} is not supported. Choose from 'unet', 'pspnet', 'deeplab'."
        )
    return model


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
    sanity_check=False,
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_meanIoU": [],
        "tl": [],
        "vl": [],
        "viou": [],
    }
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        n_step = 0
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            n_step += 1
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / n_step)
            history["tl"].append(loss.item())

        history["train_loss"].append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        n_step = 0
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                predicted = (torch.argmax(outputs, dim=1)).long()
                n_step += 1
                val_bar.set_postfix(val_loss=val_loss / n_step)

                iou = compute_IoU(predicted, masks)
                val_iou += iou.item()
                history["vl"].append(loss.item())
                history["viou"].append(iou.item())

        history["val_meanIoU"].append(val_iou / len(val_loader))
        history["val_loss"].append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {history['train_loss'][-1]:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Mean IoU: {iou.item():.4f}"
        )

        if sanity_check:
            print("Sanity Check completed")
            return

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


def calculate_weights(data: torch.utils.data.DataLoader) -> torch.Tensor:
    class_counts = torch.zeros(7, dtype=torch.long)
    for _, mask in tqdm(data):
        for i in range(7):
            class_counts[i] += torch.sum(mask == i).item()

    print(class_counts)
    weights = 1.0 / (class_counts + 1e-3)

    weights = weights / weights.mean()
    print(weights)

    return weights


def compute_IoU(
    pred_mask: torch.tensor, mask: torch.tensor, num_classes: int = 7
) -> torch.tensor:
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_mask, mask, mode="multiclass", num_classes=num_classes
    )
    image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return image_iou
