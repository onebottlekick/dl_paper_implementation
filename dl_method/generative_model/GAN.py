import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SHAPE = (3, 32, 32)
LATENT_DIM = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.99)
EPOCHS = 200

data_loader = DataLoader(
    dataset=CIFAR10(
        root='./datasets',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(IMG_SHAPE[0])], [0.5 for _ in range(IMG_SHAPE[0])])
        ]),
        download=True
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(in_channels, out_channels, normalize=True):
            layers = [nn.Linear(in_channels, out_channels)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
            
        self.model = nn.Sequential(
            *block(LATENT_DIM, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, np.prod(IMG_SHAPE)),
            nn.Tanh()
        )
        
    def forward(self, z):
        gen_img_flat = self.model(z)
        gen_img = gen_img_flat.view(gen_img_flat.size(0), *IMG_SHAPE)
        return gen_img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(np.prod(IMG_SHAPE), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        flat_img = img.view(-1, np.prod(IMG_SHAPE))
        validity = self.model(flat_img)
        return validity

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

for epoch in range(1, EPOCHS+1):
    with tqdm(data_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for img, _ in t:
            img = img.to(DEVICE)
            
            real = torch.ones(img.size(0), 1, device=DEVICE)
            fake = torch.zeros(img.size(0), 1, device=DEVICE)            

            optimizer_G.zero_grad()
            z = torch.rand(img.size(0), LATENT_DIM, device=DEVICE)
            gen_img = generator(z)
            loss_G = criterion(discriminator(gen_img), real)
            loss_G.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(img), real)
            fake_loss = criterion(discriminator(gen_img.detach()), fake)
            loss_D = 0.5*(real_loss + fake_loss)
            loss_D.backward()
            optimizer_D.step()
            
            t.set_postfix(loss_D=f'{loss_D.item():.4f}', loss_G=f'{loss_G.item():.4f}')