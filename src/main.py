import torch
import fire
from dataset import DeepGlobeDataset
from torch.utils.data import Subset
from diceloss import DiceLoss
from utils import train_and_validate, get_model
from typing import TypedDict
from visualization import after_training_plots
import segmentation_models_pytorch as smp


class StandartConfig(TypedDict):
    data_dir: str = "../dataset/train"
    val_size: float = 0.2
    test_size: float = 0.1
    transform = None
    target_transform = None
    batch_size: int = 1
    learning_rate: float = 2e-4
    epochs: int = 5
    encoder_name: str = "resnet18"
    encoder_weights: str = "imagenet"
    activation: str = "logsoftmax"
    in_channels: int = 3
    classes: int = 7
    device: str = "mps"
    checkpoints_dir: str = "./checkpoints"
    freeze_encoder_layers: int = -2  # freeze encoder excluding 2 last layers


def main(model_name: str = "unet", cfg=None, load_from_checkpoint=None):
    if cfg is None:
        cfg = StandartConfig

    model = get_model(model_name, cfg)
    if load_from_checkpoint is not None:
        model.load_state_dict(
            torch.load(load_from_checkpoint, map_location=torch.device(cfg.device))
        )
    model.to(cfg.device)
    if cfg.freeze_encoder_layers:
        print("freezing encoder layers")
        for layer in list(model.encoder.children())[: cfg.freeze_encoder_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    train_dataset = DeepGlobeDataset(
        data_dir=cfg.data_dir,
        split="train",
        val_size=cfg.val_size,
        transform=cfg.transform,
        target_transform=cfg.target_transform,
    )
    val_dataset = DeepGlobeDataset(
        data_dir=cfg.data_dir,
        split="val",
        val_size=cfg.val_size,
        transform=cfg.transform,
        target_transform=cfg.target_transform,
    )

    weights = torch.tensor([0.0401, 0.0074, 0.0513, 0.0357, 0.1287, 0.0499, 6.6869]).to(
        cfg.device
    )
    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    print("Sanity Check")
    train_and_validate(
        model=model,
        train_loader=torch.utils.data.DataLoader(
            Subset(train_dataset, indices=(0, 1)),
            batch_size=cfg.batch_size,
            shuffle=True,
        ),
        val_loader=torch.utils.data.DataLoader(
            Subset(val_dataset, indices=(0, 1)),
            batch_size=cfg.batch_size,
            shuffle=False,
        ),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg.learning_rate),
        criterion=loss_fn,
        epochs=cfg.epochs,
        device=cfg.device,
        model_name=model_name,
        checkpoint_dir=cfg.checkpoints_dir,
        sanity_check=True,
    )

    model, history = train_and_validate(
        model=model,
        train_loader=torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True
        ),
        val_loader=torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False
        ),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg.learning_rate),
        criterion=loss_fn,
        epochs=cfg.epochs,
        device=cfg.device,
        model_name=model_name,
        checkpoint_dir=cfg.checkpoints_dir,
    )

    after_training_plots(history, save_path="auto")

    return model, history


if __name__ == "__main__":
    fire.Fire(main)
