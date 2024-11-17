import os
import torch
import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from dev.data.datasets.supervised_dataset import SupervisedReconstructionDataset
from dev.data.readers.png_reader import PNGReader
from dev.data.readers.tiff_reader import TiffReader
from dev.transforms.transform import _Transform

class Padding(_Transform):
    """
    This class is used to apply reflective padding to images, ensuring that they have a specific height and width. 
    It is especially useful when the input images have variable dimensions and need to be adjusted to a fixed size without distorting the original information.
    """
    def __init__(self, target_h_size: int, target_w_size: int):
        """
        ----------
        Parameters:
            target_h_size: Target height for images after padding.
            target_w_size: Target width for images after padding.
        """
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        ----------
        Parameters:
            Input image as a NumPy array (np.ndarray). Can be grayscale (2D) or with color channels (3D).
        """
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)

        # Reflective padding on height and width dimensions
        if x.ndim == 2:  # Grayscale image
            x = np.expand_dims(x, axis=2)  # Agrega canal de dimensión 1
        padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        # Convert to PyTorch Tensor and reorder to (channels, height, width)
        padded = torch.from_numpy(np.transpose(padded, (2, 0, 1))).float()
        
        return padded

class F3DataModule(L.LightningDataModule):
    """
    This class is a custom implementation of a PyTorch Lightning data module designed to load and process image and annotation data for supervised tasks. 
    It is specifically configured to work with data in TIFF and PNG formats, which is common in seismic segmentation applications, such as in your project with the F3 Dataset.
    """
    def __init__(
        self,
        train_path: str,
        annotations_path: str,
        transforms: _Transform = None,
        batch_size: int = 1,
        num_workers: int = None,
    ):
        """
        Parameters
        ----------
            train_path: Path to the training, validation and test images directory.
            annotations_path: Path to the corresponding annotations directory.
            transforms: Optional transformations to be applied to the images.
            batch_size: Batch size for dataloaders.
            num_workers: Number of threads to load data (defaults to the number of CPU cores).
        """
        super().__init__()
        self.train_path = Path(train_path)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.datasets = {}

    def setup(self, stage=None):
        """
        Configure training, validation, testing, or prediction datasets depending on the stage.
        """
        if stage == "fit":
            # Training
            train_img_reader = TiffReader(self.train_path / "train") # Use TiffReader for TIFF images and PNGReader.
            train_label_reader = PNGReader(self.annotations_path / "train") # Use PNGReader for PNG annotations.
            train_dataset = SupervisedReconstructionDataset(
                readers=[train_img_reader, train_label_reader],
                transforms=self.transforms,
            ) # Combines images and annotations using the Supervised Reconstruction Dataset class, which associates images with their corresponding labels.
            
            # Validation
            val_img_reader = TiffReader(self.train_path / "val")
            val_label_reader = PNGReader(self.annotations_path / "val")
            val_dataset = SupervisedReconstructionDataset(
                readers=[val_img_reader, val_label_reader],
                transforms=self.transforms,
            ) # Combines images and annotations using the Supervised Reconstruction Dataset class, which associates images with their corresponding labels.

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            # Testing
            test_img_reader = TiffReader(self.train_path / "test")
            test_label_reader = PNGReader(self.annotations_path / "test")
            test_dataset = SupervisedReconstructionDataset(
                readers=[test_img_reader, test_label_reader],
                transforms=self.transforms,
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.
        ----------
        Parameters:
            batch_size: Number of images per batch.
            shuffle=True: It randomly shuffles the images in each epoch, which is important for good training.
            num_workers: Number of threads used to load the data.
        """
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        ----------
        Parameters:
            batch_size: Number of images per batch.
            shuffle=False: It does not mix the data, since it is not necessary.
            num_workers: Number of threads used to load the data.
        """
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
