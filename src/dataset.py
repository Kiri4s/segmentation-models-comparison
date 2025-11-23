import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepGlobeDataset(Dataset):
    """
    A PyTorch Dataset for the DeepGlobe Land Cover Classification challenge.
    Assumes the dataset is in a directory with flat structure, containing images
    and corresponding masks.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
    ):
        """
        Initializes the DeepGlobeDataset.
        Args:
            data_dir (str): Directory path containing the images and masks.
            transform (Optional[Callable]): A transform to apply to the images.
        """
        self.data_dir = data_dir
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.endswith("_sat.jpg")]
        )
        self.masks = sorted(
            [f for f in os.listdir(data_dir) if f.endswith("_mask.png")]
        )

        self.transform = (
            transforms.Compose([transforms.ToTensor()])
            if transform is None
            else transform
        )
        self.target_transform = transforms.Compose([transforms.ToTensor()])

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
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        mask_array = np.array(mask)
        # Assuming the mask is an RGB image, we convert RGB values to class labels.
        # This mapping might need adjustment based on the actual dataset's color-to-class encoding.
        class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
        class_mapping = {
            (0, 255, 255): 0,  # Urban land
            (255, 255, 0): 1,  # Agriculture land
            (255, 0, 255): 2,  # Rangeland
            (0, 255, 0): 3,  # Forest land
            (255, 0, 0): 4,  # Water
            (255, 255, 255): 5,  # Barren land
            (0, 0, 0): 6,  # Unknown
        }
        for rgb, label in class_mapping.items():
            class_mask[(mask_array == rgb).all(axis=-1)] = label

        if self.target_transform:
            class_mask = self.target_transform(class_mask).squeeze(0)
        return image, class_mask
