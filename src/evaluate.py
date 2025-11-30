import torch
import fire
from dataset import DeepGlobeDataset
from torchmetrics.classification import MulticlassConfusionMatrix
import os
import pandas as pd
from tqdm import tqdm
from diceloss import DiceLoss
from main import StandartConfig
from utils import get_model
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


MODELS_LIST = [
    "unet_Epochs_4_lf_DiceLoss_lr_0.0002",
    # "pspnet_Epochs:5",
    "fpn_Epochs:5_lf:DiceLoss_lr:0.0002",
    "pspnet_Epochs:5_lf:DiceLoss_lr:0.0002",
]


def evaluate(model_list=MODELS_LIST, cfg=None):
    if cfg is None:
        cfg = StandartConfig

    criterion = smp.losses.DiceLoss(
        smp.losses.MULTICLASS_MODE, from_logits=True
    )  # torch.nn.CrossEntropyLoss()
    conf_matrix = MulticlassConfusionMatrix(num_classes=cfg.classes).to(cfg.device)

    results = []

    for model_file in os.listdir(cfg.checkpoints_dir):
        if not model_file.endswith(".pth"):
            continue

        model_name = model_file.split("_best.pth")[0]
        if model_name not in model_list:
            continue
        model_path = os.path.join(cfg.checkpoints_dir, model_file)

        model = get_model(model_name.split("_")[0], cfg)

        test_dataset = DeepGlobeDataset(
            data_dir=cfg.data_dir,
            split="test",
            test_size=cfg.test_size,
            transform=cfg.transform,
            target_transform=cfg.target_transform,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False
        )
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(cfg.device))
        )
        model.to(cfg.device)
        model.eval()

        total_loss = 0
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
        plot_confusion_matrix(
            cm,
            save_path=f"../results/{model_name}_confmatrix.png",
            normalization="row",
            model_name=model_name.split("_")[0],
        )
        conf_matrix.reset()

    results_df = pd.DataFrame(results)
    print("\n--- Comparison ---")
    print(results_df)


def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: list = [
        "Urban land",
        "Agriculture land",
        "Rangeland",
        "Forest land",
        "Water",
        "Barren land",
        "Unknown",
    ],
    save_path: str = "confusion_matrix.png",
    normalization: str = "total",
    model_name: str = "",
) -> None:
    """
    Plots and saves the confusion matrix.

    Args:
        cm (torch.Tensor): Confusion matrix tensor.
        class_names (list): List of class names.
        save_path (str): The path to save the plot image.
    """
    if normalization == "total":
        cm = cm / cm.sum()
    elif normalization == "row":
        cm = cm / cm.sum(dim=1, keepdim=True)
    elif normalization == "column":
        cm = cm / cm.sum(dim=0, keepdim=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm.cpu().numpy(),
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"{model_name} confusion matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(evaluate)
