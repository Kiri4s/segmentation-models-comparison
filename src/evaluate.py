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


def evaluate(model_list=["pspnet"], cfg=None):
    if cfg is None:
        cfg = StandartConfig

    test_dataset = DeepGlobeDataset(
        data_dir=cfg.data_train,
        split="val",
        val_size=cfg.val_size,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    criterion = DiceLoss()  # torch.nn.CrossEntropyLoss()
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
