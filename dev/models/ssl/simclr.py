import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn

class SimCLRModel(L.LightningModule):
    """
    This class is a example of self-supervised learning, where the goal is to learn robust representations 
    by comparing augmented views of the same images (positive contrast) and distinguishing them from other images 
    in the batch (negative contrast). 
    """
    def __init__(self, encoder: nn.Module, projection_dim=128, learning_rate=1e-3):
        """
        ----------
        Parameters:
            encoder: A model that acts as a feature extractor (e.g., the ResNetEncoder).
            projection_dim: Dimension of the projected space, where contrastive comparisons occur.
            learning_rate: Learning rate for the optimizer.
        """
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        ) # A fully connected network with two linear layers and an intermediate ReLU activation.
        # It reduces the features to a low-dimensional space (projection_dim), which is important for the computation of the contrastive loss.
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Pass the images through the encoder to obtain a feature tensor.
        """
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        x, y = batch # We assume that batch contains two elements: images and labels
        z = self(x) # We apply the encoder and projection to the images
        # We proceed to calculate the loss
        loss = self.nt_xent_loss(z, z) # We use the same projections for now
        self.log("train_loss", loss)  # Save the loss for checkpoint recovery
        return loss

    def nt_xent_loss(self, z1, z2, temperature=0.5):
        """
        Calculates the temperature normalized cross entropy loss (NT-Xent Loss), which is fundamental in SimCLR.
        """
        # Normalize the representations before calculating the loss
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.t()) / temperature
        labels = torch.arange(z1.size(0)).to(self.device) 
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        """
        Using the Adam optimizer to update weights
        """
        return Adam(self.parameters(), lr=self.learning_rate)
