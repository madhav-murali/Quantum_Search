import torch
import torch.nn as nn
import torchvision.models as models
import timm
from .lenet_cnn import LeNetCNN
from .lenet_improved import LeNet5Quantum, LeNetCNNImproved

class BackboneFactory:
    """
    Factory to create extraction backbones.
    """
    @staticmethod
    def create(name, pretrained=True, in_channels=3):
        """
        Creates a backbone model.
        Args:
            name (str): 'resnet18', 'resnet50', 'vit_base', 'lenet', 'lenet5', 'lenet_improved'
            pretrained (bool): Whether to use pretrained weights.
            in_channels (int): Number of input channels (3 for RGB, 13 for All).
        Returns:
            nn.Module: The feature extractor.
            int: The output feature dimension.
        """
        name = name.lower()
        
        if 'resnet' in name:
            if name == 'resnet18':
                model = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif name == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            else:
                raise ValueError(f"Unknown ResNet: {name}")
            
            # Modify first layer if input channels != 3
            if in_channels != 3:
                original_layer = model.conv1
                model.conv1 = nn.Conv2d(
                    in_channels, 
                    original_layer.out_channels,
                    kernel_size=original_layer.kernel_size,
                    stride=original_layer.stride,
                    padding=original_layer.padding,
                    bias=original_layer.bias
                )
            
            # Remove the classification head (fc)
            model.fc = nn.Identity()
            return model, feature_dim
            
        elif 'vit' in name:
            # Using timm for ViT as it's more flexible
            # vit_base_patch16_224
            model_name = 'vit_base_patch16_224'
            model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_channels, num_classes=0)
            feature_dim = model.num_features
            return model, feature_dim
        
        elif name == 'lenet5':
            # Classic LeNet-5 optimized for quantum (84 features)
            model = LeNet5Quantum(in_channels=in_channels, input_size=64, use_batchnorm=True, dropout_rate=0.3)
            feature_dim = model.feature_dim
            return model, feature_dim
        
        elif name == 'lenet_improved':
            # Improved LeNet with BatchNorm and Dropout
            model = LeNetCNNImproved(in_channels=in_channels, input_size=64, dropout_rate=0.25)
            feature_dim = model.feature_dim
            return model, feature_dim
        
        elif name == 'lenet':
            # Original LeNet-like CNN - no pretrained weights available
            model = LeNetCNN(in_channels=in_channels, input_size=64)
            feature_dim = model.feature_dim
            return model, feature_dim

        else:
            raise ValueError(f"Unknown backbone: {name}")


