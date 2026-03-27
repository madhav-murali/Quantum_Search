import torch
import torch.nn as nn

class LeNetCNN(nn.Module):
    """
    LeNet-like CNN with 3 convolutional layers.
    
    Architecture:
        - Conv1: in_channels -> 32 filters, 5x5 kernel, ReLU, 2x2 MaxPool
        - Conv2: 32 -> 64 filters, 5x5 kernel, ReLU, 2x2 MaxPool
        - Conv3: 64 -> 128 filters, 3x3 kernel, ReLU, 2x2 MaxPool
        - Flatten: Convert to 1D feature vector
    
    Args:
        in_channels (int): Number of input channels (3 for RGB, 13 for multispectral)
        input_size (int): Expected input image size (height/width, assumes square images)
    """
    
    def __init__(self, in_channels=3, input_size=64):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Calculate feature dimension after convolutions and pooling
        self.feature_dim = self._calculate_feature_dim()
        
    def _calculate_feature_dim(self):
        """Calculate the flattened feature dimension by doing a forward pass."""
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = self.flatten(x)
            return x.shape[1]
    
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Flattened features of shape (B, feature_dim)
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        return x
