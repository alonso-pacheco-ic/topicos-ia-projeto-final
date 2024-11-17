import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from dev.data.datasets.simclr_dataset import SimCLRTransform
from dev.data.datasets.simclr_dataset import SimCLRDataset

class TinyImageNetDataModule:
    """
    This class is designed to manage the data pipeline for training and validating a model using the Tiny ImageNet dataset.
    Standard approach in PyTorch Lightning, where data handling is separated into a module to keep the code organized and reusable.
    """
    def __init__(self, data_dir, batch_size, num_workers=4):
        """
        ----------
        Parameters:
            data_dir: Directory where the data will be stored
            batch_size: Batch size for training and validation.
            num_workers: Number of threads used by the DataLoader to load data (default 4).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = SimCLRTransform() # Use the SimCLR transformation

    def setup(self, stage=None):
        """
        Define the paths to the training and validation data:
        1) Assume that the training images are in a folder called train.
        2) The validation images are in a folder called val
        """
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        # Applies the transformations defined in SimCLRTransform to the dataset to generate the required views.
        self.train_dataset = SimCLRDataset(ImageFolder, train_dir, transform=self.transform) 
        self.val_dataset = SimCLRDataset(ImageFolder, val_dir, transform=self.transform)
        # ImageFolder: Standard structure where images are organized into folders by class.

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
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        ----------
        Parameters:
            batch_size: Number of images per batch.
            shuffle=False: It does not mix the data, since it is not necessary in validation.
            num_workers: Number of threads used to load the data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )