import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

IMG_SIZE = 32
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 64
LATENT_DIM = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
EPOCHS = 200

dataset = datasets.MNIST(
    root='./datasets',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
    ])
)

data_loader = DataLoader(
    dataset = dataset,
    shuffle=True,
    batch_size=BATCH_SIZE
)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128*self.init_size**2))
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, IMG_CHANNELS, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.model(out)
        return img
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(in_filters, out_filters, normalize=True):
            layer = [
                nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
                ]
            if normalize:
                layer.append(nn.BatchNorm2d(out_filters, 0.8))
            return layer
        
        self.model = nn.Sequential(
            *block(IMG_CHANNELS, 16, normalize=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128)
        )
        
        ds_size = IMG_SIZE // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128*ds_size**2, 1),
            nn.Sigmoid()
            )
        
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    

adversarial_criterion = nn.BCELoss().to(DEVICE)

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

for epoch in range(EPOCHS):
    with tqdm(data_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for i, (img, _) in enumerate(t):
            valid = torch.ones(img.size(0), 1, requires_grad=False).to(DEVICE)
            fake = torch.zeros(img.size(0), 1, requires_grad=False).to(DEVICE)
            
            real_img = img.to(DEVICE)
            
            optimizer_G.zero_grad()
            z = torch.rand(img.shape[0], LATENT_DIM).to(DEVICE)
            gen_img = generator(z)
            g_loss = adversarial_criterion(discriminator(gen_img), valid)
            g_loss.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            real_loss = adversarial_criterion(discriminator(real_img), valid)
            fake_loss = adversarial_criterion(discriminator(gen_img.detach()), fake)
            d_loss = (real_loss+fake_loss)/2
            d_loss.backward()
            optimizer_D.step()