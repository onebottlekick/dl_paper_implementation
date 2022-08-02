import torch
from torch import nn


# assume that image_shape = (B, 3, 224, 224) = (B, C, H, W)
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), # (48, 55, 55)
            self.relu,
            nn.Conv2d(48, 128, kernel_size=5, padding=2), # (128, 55, 55)
            self.relu,
            self.max_pool, # (128, 27, 27)
            nn.Conv2d(128, 192, kernel_size=3, padding=1), # (192, 27, 27)
            self.relu,
            self.max_pool, # (192, 13, 13)
            nn.Conv2d(192, 192, kernel_size=3, padding=1), # (192, 13, 13)
            self.relu,
            nn.Conv2d(192, 128, kernel_size=3, padding=1), # (128, 13, 13)
            nn.Flatten(), # 21632
            nn.Linear(21632, 2048), # 2048
            self.relu,
            nn.Linear(2048, 2048), # 2048
            self.relu,
            nn.Linear(2048, 1000) # 1000      
        )
        
        
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':    
    model = AlexNet()
    x = torch.rand(1, 3, 224, 224)
    assert model(x).shape == (1, 1000)