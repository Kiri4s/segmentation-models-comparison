import matplotlib.pyplot as plt
from typing import Dict, List


def plot_loss(
    history: Dict[str, List[float]], save_path: str = "loss_plot.png"
) -> None:
    """
    Plots the training and validation loss over epochs.

    Args:
        history (Dict[str, List[float]]): A dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str): The path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_iou(history: Dict[str, List[float]], save_path: str = "iou_plot.png") -> None:
    """
    Plots the validation Intersection over Union (IoU) over epochs.

    Args:
        history (Dict[str, List[float]]): A dictionary containing 'val_accuracy' list.
        save_path (str): The path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["val_accuracy"], label="Validation IoU")
    plt.title("Validation IoU over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
