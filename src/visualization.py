import matplotlib.pyplot as plt
from typing import Dict
import json


def plot_per_epoch_loss(
    history: Dict, save_path: str = "loss_per_epoch_plot.png", ax=None
) -> None:
    """
    Plots the training and validation loss over epochs.

    Args:
        history (Dict): A dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str): The path to save the plot image.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_per_step_loss(
    history: Dict, save_path: str = "loss_per_step_plot.png", ax=None
) -> None:
    """
    Plots the training and validation loss over steps.

    Args:
        history (Dict): A dictionary containing 'tl' and 'vl' lists.
        save_path (str): The path to save the plot image.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(history["tl"], label="Train Loss")
    ax.plot(history["vl"], label="Validation Loss")
    ax.set_title("Loss over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_per_epoch_iou(history: Dict, save_path: str = "iou_per_epoch_plot.png", ax=None) -> None:
    """
    Plots the validation Intersection over Union (IoU) over epochs.

    Args:
        history (Dict): A dictionary containing 'val_meanIoU' list.
        save_path (str): The path to save the plot image.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(history["val_meanIoU"], label="Validation IoU")
    ax.set_title("Validation IoU over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU Score")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
    return fig
    
    
def plot_per_step_iou(history: Dict, save_path: str = "iou_per_step_plot.png", ax=None) -> None:
    """
    Plots the validation Intersection over Union (IoU) over steps.

    Args:
        history (Dict): A dictionary containing 'viou' list.
        save_path (str): The path to save the plot image.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(history["viou"], label="Validation IoU")
    ax.set_title("Validation IoU over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("IoU Score")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
    return fig


def after_training_plots(history: Dict, save_path: str = "auto") -> None:
    """
    Generates and saves loss and IoU plots after training.
    """
    if save_path == "auto":
        try:
            save_path = f"../results/{history['metadata']}_train_plots"
        except KeyError:
            save_path = "../results/unnamed_train_plots"

    # Per epoch plots
    fig_epoch, axes_epoch = plt.subplots(1, 2, figsize=(20, 8))
    fig_epoch.suptitle('Per Epoch Training Analysis', fontsize=16)
    plot_per_epoch_loss(history, save_path=None, ax=axes_epoch[0])
    plot_per_epoch_iou(history, save_path=None, ax=axes_epoch[1])
    if save_path:
        plt.savefig(save_path + "_per_epoch.png")
    plt.close(fig_epoch)

    # Per step plots
    fig_step, axes_step = plt.subplots(1, 2, figsize=(20, 8))
    fig_step.suptitle('Per Step Training Analysis', fontsize=16)
    plot_per_step_loss(history, save_path=None, ax=axes_step[0])
    plot_per_step_iou(history, save_path=None, ax=axes_step[1])
    if save_path:
        plt.savefig(save_path + "_per_step.png")
    plt.close(fig_step)
    

if __name__ == "__main__":
    with open("./checkpoints/pspnet_Epochs:5_training_history.json", "r") as f:
        history = json.load(f)
    #plot_per_epoch_loss(history, save_path="../results/pspnet_loss_per_epoch_plot.png")
    #plot_per_epoch_iou(history, save_path="../results/pspnet_iou_per_epoch_plot.png")
    after_training_plots(history, save_path="../results/pspnet_training_summary.png")
