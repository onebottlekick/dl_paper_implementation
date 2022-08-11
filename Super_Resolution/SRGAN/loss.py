import torch
import torch.nn as nn
from torchvision.models import vgg19

from model import Discriminator
    
    
class VGGLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = 'cpu' if args.cpu else 'cuda'
        if args.vgg_type == 22:
            self.vgg = vgg19(pretrained=True).features[:9].eval().to(self.device)
        elif args.vgg_type == 54:
            self.vgg = vgg19(pretrained=True).features[:36].eval().to(self.device)
            
        self.loss = nn.MSELoss()
        
        for param in self.vgg.parameters():
            param.require_grad = False
            
    def forward(self, x, y):
        x = self.vgg(x)
        y = self.vgg(y)
        
        return self.loss(x, y)
    
    
class AdversarialLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = 'cpu' if args.cpu else 'cuda'
        self.discriminator = Discriminator(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.bce_loss = nn.BCELoss()
        
    def forward(self, gen, real):
        gen = gen.detach()
        
        self.optimizer.zero_grad()
        d_fake = self.discriminator(gen)
        d_real = self.discriminator(real)
        
        valid = torch.ones(real.shape[0], 1).to(self.device)
        fake = torch.zeros(real.shape[0], 1).to(self.device)
        real_loss = self.bce_loss(d_real, valid)
        fake_loss = self.bce_loss(d_fake, fake)
        d_loss = (real_loss + fake_loss) / 2.
        
        d_loss.backward()
        self.optimizer.step()
        
        g_loss = self.bce_loss(self.discriminator(gen), valid)
        
        return g_loss
    
    
if __name__ == '__main__':
    vgg = vgg19(pretrained=True).features
    print(vgg)