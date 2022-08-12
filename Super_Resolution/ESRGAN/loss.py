import torch
import torch.nn as nn
from torchvision.models import vgg19

from model import Discriminator
    
    
class VGGLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = 'cpu' if args.cpu else 'cuda'
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(self.device)            
        self.loss = nn.MSELoss()
        
        for param in self.vgg.parameters():
            param.require_grad = False
            
    def forward(self, x, y):
        x = self.vgg(x)
        y = self.vgg(y)
        
        return self.loss(x, y)
    

# TODO implement relativistic gan loss
class AdversarialLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = 'cpu' if args.cpu else 'cuda'
        self.discriminator = Discriminator(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.bce_loss = nn.BCELoss()
        
    def forward(self, gen, real):        
        pass
    
    
if __name__ == '__main__':
    vgg = vgg19(pretrained=True).features
    print(vgg)