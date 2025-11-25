import torch
import fire
from dataset import DeepGlobeDataset
from typing import TypedDict
import segmentation_models_pytorch as smp
from torchvision import transforms as T
from torchmetrics.classification import MulticlassConfusionMatrix
import os
import pandas as pd
from tqdm import tqdm


class StandartConfig(TypedDict):
    data_test: str = "../dataset/train"
    val_size: float = 0.2
    transform = None
    batch_size: int = 1
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    activation: str = "logsoftmax"
    in_channels: int = 3
    classes: int = 7
    device: str = "mps"
    checkpoints_dir: str = "./checkpoints"


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
    return model


def evaluate(model_list=["pspnet"], cfg=None):
    if cfg is None:
        cfg = StandartConfig

    test_dataset = DeepGlobeDataset(
        data_dir=cfg.data_test,
        split="val",
        val_size=cfg.val_size,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    conf_matrix = MulticlassConfusionMatrix(num_classes=cfg.classes).to(cfg.device)

    results = []

    for model_file in os.listdir(cfg.checkpoints_dir):
        if not model_file.endswith(".pth"):
            continue

        model_name = model_file.split("_best.pth")[0]
        if model_name not in model_list:
            continue
        model_path = os.path.join(cfg.checkpoints_dir, model_file)

        model = get_model(model_name, cfg)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(cfg.device))
        )
        model.to(cfg.device)
        model.eval()

        total_loss = 0
        total_iou = 0
        print(f"Evaluating model: {model_name}")
        with torch.no_grad():
            for images, masks in tqdm(test_loader):
                images, masks = images.to(cfg.device), masks.to(cfg.device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                conf_matrix.update(predicted, masks)

        avg_loss = total_loss / len(test_loader)

        cm = conf_matrix.compute()
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag())
        mean_iou = iou.nanmean()

        results.append(
            {"model": model_name, "loss": avg_loss, "mean_iou": mean_iou.item()}
        )

        print(f"Model: {model_name}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Confusion Matrix:\n{cm.cpu().numpy()}")

        conf_matrix.reset()

    results_df = pd.DataFrame(results)
    print("\n--- Comparison ---")
    print(results_df)


if __name__ == "__main__":
    fire.Fire(evaluate)
