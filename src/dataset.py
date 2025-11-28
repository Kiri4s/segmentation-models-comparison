import os
import torch
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DeepGlobeDataset(Dataset):
    """
    A PyTorch Dataset for the DeepGlobe Land Cover Classification challenge.
    Assumes the dataset is in a directory with flat structure, containing images
    and corresponding masks.
    """

    class_mapping = {
        (  0, 255, 255): 0,  # Urban land
        (255, 255,   0): 1,  # Agriculture land
        (255,   0, 255): 2,  # Rangeland
        (  0, 255,   0): 3,  # Forest land
        (  0,   0, 255): 4,  # Water
        (255, 255, 255): 5,  # Barren land
        (  0,   0,   0): 6,  # Unknown
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_size: float = 0.2,
        seed: int = 42,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initializes the DeepGlobeDataset.
        Args:
            data_dir (str): Directory path containing the images and masks.
            transform (Optional[Callable]): A transform to apply to the images.
        """
        images = sorted([f for f in os.listdir(data_dir) if f.endswith("_sat.jpg")])
        masks = sorted([f for f in os.listdir(data_dir) if f.endswith("_mask.png")])
        train_images, val_images, train_masks, val_masks = train_test_split(
            images,
            masks,
            test_size=val_size,
            random_state=seed,
        )
        self.data_dir = data_dir

        if split == "full":
            self.images = images
            self.masks = masks
        elif split == "train":
            self.images = train_images
            self.masks = train_masks
        elif split == "val":
            self.images = val_images
            self.masks = val_masks
        else:
            raise ValueError("unsupported split")

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a sample from the dataset at the specified index.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            A tuple containing the image and its corresponding mask.
        """
        img_path = os.path.join(self.data_dir, self.images[idx])
        mask_path = os.path.join(self.data_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask_array = np.array(mask)

        # Assuming the mask is an RGB image, we convert RGB values to class labels.
        # This mapping might need adjustment based on the actual dataset's color-to-class encoding.
        class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)

        for rgb, label in self.class_mapping.items():
            class_mask[(mask_array == rgb).all(axis=-1)] = label

        class_mask = torch.from_numpy(class_mask).long()
        if self.target_transform:
            class_mask = self.target_transform(class_mask.unsqueeze(0)).squeeze(0)
        return image, class_mask

    @staticmethod
    def label_to_rgb_mask(label_mask: np.ndarray) -> Image.Image:
        """
        Converts a 2D label mask to a 3D RGB mask.
        Args:
            label_mask (np.ndarray): A 2D numpy array where each pixel is a class label.
        Returns:
            PIL.Image.Image: An RGB image representing the mask.
        """
        rgb_mask = np.zeros(label_mask.shape + (3,), dtype=np.uint8)
        reverse_mapping = {
            label: rgb for rgb, label in DeepGlobeDataset.class_mapping.items()
        }
        for label, rgb in reverse_mapping.items():
            rgb_mask[label_mask == label] = rgb
        return Image.fromarray(rgb_mask)
