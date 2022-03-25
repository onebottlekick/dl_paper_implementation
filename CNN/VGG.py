import torch
from torch import nn


model_config = {
    '11':[(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
    '13':[(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
    '16':[(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
    '19':[(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)]
}

class VGG(nn.Module):
    def __init__(self, config):
        super(VGG, self).__init__()
        
        self.in_channels = 3
        
        def create_block(out_channels, repeat):
            layers = []
            for _ in range(repeat):
                layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                self.in_channels = out_channels
            return layers
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.fc = [
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        ]
        
        self.layers = []
        
        for conf in config:
            self.layers += create_block(conf[0], conf[1])
            self.layers.append(self.maxpool)                
        self.layers += self.fc
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = VGG(model_config['16'])
    x = torch.rand(1, 3, 224, 224)
    assert model(x).shape == (1, 1000)