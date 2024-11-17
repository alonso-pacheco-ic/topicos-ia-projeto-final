import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import Accuracy

class UNet(nn.Module):
    """
    This class implements the classical U-Net architecture, which is widely used in semantic segmentation tasks. 
    The model follows an "encoder-decoder" design with skip connections that allow the transfer of high-resolution 
    information from the initial encoder layers to the corresponding decoder layers.
    
    Metrics used
        1) JaccardIndex (mIoU):
        Evaluates the average intersection over union (IoU) between predictions and true labels.
        It is common in semantic segmentation, especially to evaluate spatial accuracy.
        
        2) MulticlassAccuracy (MCA):
        Calculates the average accuracy per class, useful for balancing uneven datasets.
        
        3) Accuracy (PA):
        Overall accuracy based on all correctly classified pixels.
    """
    def __init__(self, in_channels=3, num_classes=6):
        """
        ----------
        Parameters:
            in_channels: Number of channels in the input images. Default is 3 (RGB).
            num_classes: Number of output classes. Each class corresponds to a value in the segmentation map.
        """
        super(UNet, self).__init__()

        # The model is divided into three parts: encoder, bottleneck and decoder.
        
        ###
        # ENCODER: progressively reduces spatial resolution as feature depth increases:
        ###
        
        # Two convolutions (with ReLU activations) process the input.
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Max pooling reduces the spatial resolution by a factor of 2 (downsampling).
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Similar to the first one, but with more filters (128).
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Pooling reduces the resolution even further.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ###
        # BOTTLENECK: Processes the most compressed features
        ###
        
        # Two deep convolutions with 256 filters capture abstract and higher-level information.
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        ###
        # DECODER: Restores spatial resolution using transposed convolutions and combines information from skip connections
        ###
        
        # Performs upsampling (doubles spatial resolution).
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        # The bottleneck output is combined with the encoder features (x2), using skip connections.
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Similar to the first layer of the decoder.
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        ) # The last layer generates an output map with dimensions [batch_size, num_classes, height, width]. 

    def forward(self, x):
        """
        1) Generates initial features.
        2) Extracts compressed features.
        3) Combines with encoder features (skip connection).
        4) Combines with initial encoder features.
        """
        # Encoder
        x1 = self.encoder1(x)
        x1_p = self.pool1(x1)
        x2 = self.encoder2(x1_p)
        x2_p = self.pool2(x2)

        # Middle
        x_middle = self.middle(x2_p)

        # Decoder
        x = self.upconv2(x_middle)
        x = self.decoder2(x + x2)

        x = self.upconv1(x)
        x = self.decoder1(x + x1) 

        return x # The output is in the form [batch_size, num_classes, height, width], where each channel corresponds to a class.

class UNetLightning(L.LightningModule):
    """
    This class implements a U-Net-based segmentation model using the PyTorch Lightning library to simplify training, validation, and testing. 
    This class integrates with common metrics for semantic segmentation and an optimizer to update model parameters.
    """
    def __init__(self, in_channels=1, num_classes=6, learning_rate=1e-3):
        """
        ----------
        Parameters:
            in_channels: Number of channels in the input images. Default is 3 (RGB).
            num_classes: Number of output classes. Each class corresponds to a value in the segmentation map.
            learning_rate: Optimizer learning rate.
        """
        super(UNetLightning, self).__init__()
        self.model = UNet(in_channels, num_classes)
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Métrics
        self.train_miou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_miou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.test_miou = JaccardIndex(task="multiclass", num_classes=num_classes)

        self.train_mca = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_mca = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.test_mca = MulticlassAccuracy(num_classes=num_classes, average='macro')

        self.train_pa = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        self.val_pa = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        self.test_pa = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        
        # Lists for losses
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        """
        This method simply calls the U-Net model to perform prediction on the input data.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Preprocessing: Labels are squeezed(1) to remove extra dimensions and converted to long type (needed for F.cross_entropy).
        Prediction: One pass through the model (outputs = self(images)).
        Loss calculation: Uses F.cross_entropy, which combines the softmax function and cross-entropy loss.
        """
        images, labels = batch
        labels = labels.squeeze(1).long()
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        # Metrics calculation
        train_miou = self.train_miou(outputs.softmax(dim=1), labels)
        train_mca = self.train_mca(outputs.softmax(dim=1), labels)
        train_pa = self.train_pa(outputs.softmax(dim=1), labels)

        # Save the losses
        self.train_losses.append(loss.item())

        # Record metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_mIoU', train_miou, on_step=False, on_epoch=True)
        self.log('train_MCA', train_mca, on_step=False, on_epoch=True)
        self.log('train_PA', train_pa, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Similar to training step. Do not optimize the model, they only calculate metrics and losses.
        """
        images, labels = batch
        labels = labels.squeeze(1).long()
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        # Metrics calculation
        val_miou = self.val_miou(outputs.softmax(dim=1), labels)
        val_mca = self.val_mca(outputs.softmax(dim=1), labels)
        val_pa = self.val_pa(outputs.softmax(dim=1), labels)

        # Save the losses
        self.val_losses.append(loss.item())
        
        # Record metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mIoU', val_miou, on_step=False, on_epoch=True)
        self.log('val_MCA', val_mca, on_step=False, on_epoch=True)
        self.log('val_PA', val_pa, on_step=False, on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """
        Similar to training step. Do not optimize the model, they only calculate metrics and losses.
        """
        images, labels = batch
        labels = labels.squeeze(1).long()
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        # Metrics calculation
        test_miou = self.test_miou(outputs.softmax(dim=1), labels)
        test_mca = self.test_mca(outputs.softmax(dim=1), labels)
        test_pa = self.test_pa(outputs.softmax(dim=1), labels)

        # Record metrics
        self.log('test_loss', loss)
        self.log('test_mIoU', test_miou)
        self.log('test_MCA', test_mca)
        self.log('test_PA', test_pa)
        return loss

    def configure_optimizers(self):
        """
        Defines the optimizer for the model: uses Adam with the provided learning rate.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer