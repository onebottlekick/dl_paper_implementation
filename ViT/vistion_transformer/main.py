import argparse

import torch
import torch.nn as nn

from model import ViT
from utils import build_model
from configs import config_dict


parser = argparse.ArgumentParser(description='Experiment')
# parser.add_argument('data_path', metavar='DIR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='b16', help='model architecture (default: b16)')
parser.add_argument('-e', '--epochs', default=100, type=int, help='number of epochs (default: 100)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-o', '--optimizer', default='adam', type=str, metavar='OPT', help='optimizer (default: adam)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-d', '--device', default='cuda', type=str, help='device to use (default: cuda)')

best_acc1 = 0

optimizer_dict = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

'''
TODO 
- lr decay
- dataloader
'''   
def main():
    args = parser.parse_args()
    device = args.device
    train_loader = None
    val_loader = None
    
    model_config = config_dict[args.arch]
    model = build_model(ViT, model_config).to(device)
    optimizer = optimizer_dict[args.optimizer.lower()](model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, device)
        validate(val_loader, model, criterion, device)
    
'''
TODO 
- train time function
- print progress
- resume training function
'''   
def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

# TODO calc top acc
def validate(val_loader, model, criterion, device):
    model.eval()
    for i, (imgs, targets) in enumerate(val_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        

if __name__ == '__main__':
    main()