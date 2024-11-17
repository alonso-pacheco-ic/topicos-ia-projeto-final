from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

class SimCLRTransform:
    """
    This class encapsulates a specific data transformation for use with a SimCLR model with ImageneNet.
    Transformations are a series of data augmentation steps designed to generate two different but related views 
    (transformed images) of the same original image. 
    """
    def __init__(self):
        """
        ----------
        Parameters:
            None
        """
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64),  # Adjusted to 64x64 for Tiny-ImageNet
            transforms.RandomHorizontalFlip(), # Flip the image horizontally
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),  # Applies random variations of brightness, contrast, saturation and hue to the image
            transforms.RandomGrayscale(p=0.2), # Converts the image to grayscale
            transforms.ToTensor(), # Converts the format image to a PyTorch tensor (with normalized values ​​between 0 and 1)
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalización de Tiny-ImageNet # Predefined means and standard deviations
        ])

    def __call__(self, x):
        """
        Defines how the transformation is applied to an individual image.
        """
        return self.transform(x)

class SimCLRDataset(Dataset):
    """
    This class is a custom implementation of the dataset for use with the SimCLR model.
        1) Load an existing dataset.
        2) Apply SimCLR transformations to generate two distinct views of each image. These views are required for contrastive training.
    """
    def __init__(self, dataset_class, data_dir, transform=None):
        """
        ----------
        Parameters:
            dataset_class: The dataset to be used, for example, ImageFolder
            data_dir: Directory where the data will be stored
            transform: The transformations to be applied to the images
        """
        # We instantiate the dataset with the transformations not applied (to do it manually later)
        if dataset_class == ImageFolder:
            self.dataset = dataset_class(root=data_dir)  # No 'download' for ImageFolder
        else:
            self.dataset = dataset_class(root=data_dir, download=True, transform=None) # Download automatically
        self.transform = transform if transform else SimCLRTransform()  # Use SimCLRTransform if no external transformation is passed
        
    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets the image (and its label, although this is not required for SimCLR) from the loaded dataset.
        1) Positive pairs: The two views generated from the same image are considered similar.
        2) Negative pairs: All other views from other images are considered different.
        """
        # We get the image and the label (no transformation here)
        image, _ = self.dataset[idx]
        
        # Create two different views by applying transformations twice
        view1 = self.transform(image) # First transformed view 
        view2 = self.transform(image) # Second transformed view
        
        return view1, view2