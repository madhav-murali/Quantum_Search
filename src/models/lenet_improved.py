import torch
import torch.nn as nn

class LeNet5Quantum(nn.Module):
    """
    Classic LeNet-5 architecture optimized for quantum hybrid processing.
    
    This smaller architecture (84 features) works better with quantum layers
    compared to the larger LeNetCNN (8192 features).
    
    Architecture :
        - Conv1: in_channels -> 6 filters, 5x5 kernel, ReLU, 2x2 AvgPool
        - Conv2: 6 -> 16 filters, 5x5 kernel, ReLU, 2x2 AvgPool
        - FC1: 16*5*5 -> 120 units, ReLU
        - FC2: 120 -> 84 units, ReLU
    
    Args:
        in_channels (int): Number of input channels (3 for RGB, 13 for multispectral)
        use_batchnorm (bool): Whether to use batch normalization
        dropout_rate (float): Dropout rate (0 to disable)
    """
    
    def __init__(self, in_channels=3, input_size=64, use_batchnorm=True, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6) if use_batchnorm else nn.Identity()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16) if use_batchnorm else nn.Identity()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # For 32x32 input: after conv1 (28x28) -> pool1 (14x14) -> conv2 (10x10) -> pool2 (5x5)
        # For 64x64 input: after conv1 (60x60) -> pool1 (30x30) -> conv2 (26x26) -> pool2 (13x13)
        self.flatten = nn.Flatten()
        
        # Calculate dynamically
        self._setup_fc_layers(dropout_rate)
        
        self.feature_dim = 84
        
    def _setup_fc_layers(self, dropout_rate):
        """Setup FC layers based on conv output size"""
        # Test forward pass to determine size
        with torch.no_grad():
            # Use actual input size instead of hardcoded 32
            x = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.flatten(x)
            conv_output_size = x.shape[1]
        
        self.fc1 = nn.Linear(conv_output_size, 120)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through LeNet-5.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Feature vector of shape (B, 84)
        """
        # Conv layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten and FC layers
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        return x


class LeNetCNNImproved(nn.Module):
    """
    Improved LeNet-like CNN with batch normalization and dropout.
    
    Enhanced version of the original LeNetCNN with:
    - Batch normalization after each conv layer
    - Dropout for regularization
    - Better feature extraction
    
    Args:
        in_channels (int): Number of input channels
        input_size (int): Expected input size (height/width)
        dropout_rate (float): Dropout rate after pooling layers
    """
    
    def __init__(self, in_channels=3, input_size=64, dropout_rate=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.flatten = nn.Flatten()
        
        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()
        
    def _calculate_feature_dim(self):
        """Calculate the flattened feature dimension."""
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
            x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
            x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
            x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
            x = self.flatten(x)
            return x.shape[1]
    
    def forward(self, x):
        """Forward pass with batch norm and dropout."""
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.flatten(x)
        return x
