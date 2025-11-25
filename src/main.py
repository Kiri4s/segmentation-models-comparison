import torch
import fire
from dataset import DeepGlobeDataset
from utils import train_and_validate
from typing import TypedDict
import segmentation_models_pytorch as smp
from visualization import plot_loss, plot_iou
from torchvision import transforms as T


class StandartConfig(TypedDict):
    data_train: str = "../dataset/train"
    data_val: str = "../dataset/valid"
    val_size: float = 0.2
    transform = None
    batch_size: int = 1
    learning_rate: float = 1e-3
    epochs: int = 1
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    activation: str = "logsoftmax"
    in_channels: int = 3
    classes: int = 7
    device: str = "mps"
    checkpoints_dir: str = "./checkpoints"


def main(model_name: str = "unet", cfg=None):
    if cfg is None:
        cfg = StandartConfig

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
        model = smp.FPN(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            activation=cfg.activation,
        )

    elif model_name == "unet++":
        cfg.transform = T.Compose([T.Resize((2464, 2464)), T.ToTensor()])
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

    model.to(cfg.device)
    train_dataset = DeepGlobeDataset(
        data_dir=cfg.data_train,
        split="train",
        val_size=cfg.val_size,
        transform=cfg.transform,
    )
    val_dataset = DeepGlobeDataset(
        data_dir=cfg.data_train,
        split="val",
        val_size=cfg.val_size,
        transform=cfg.transform,
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
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=cfg.epochs,
        device=cfg.device,
        model_name=model_name,
        checkpoint_dir=cfg.checkpoints_dir,
    )

    plot_loss(history, save_path=f"../results/{model_name}_loss_plot.png")
    plot_iou(history, save_path=f"../results/{model_name}_iou_plot.png")

    return model, history


if __name__ == "__main__":
    fire.Fire(main)
