import torch.nn as nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self)
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(device)
        self.loss = nn.MSELoss()
        
        for param in self.vgg.parameters():
            param.require_grad = False
            
    def forward(self, x, y):
        vgg_x_features = self.vgg(x)
        vgg_y_features = self.vgg(y)
        
        return self.loss(vgg_x_features, vgg_y_features)