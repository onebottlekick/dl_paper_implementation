import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        n_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        
        return x
    
    
if __name__ == '__main__':
    model = ResNet50(10)
    torch.save(model.state_dict(), 'experiments/pretrained/hi.pth')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 10)