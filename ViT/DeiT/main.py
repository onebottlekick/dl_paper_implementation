import argparse

import torch
import torch.nn as nn

from configs import config_dict
from model import DeiT
from utils import Trainer, WarmupCosineSchedule, WarmupLinearSchedule, build_model, get_transform, get_data_loader

parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('data_path', metavar='DIR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='b16', help='model architecture (default: b16)')
parser.add_argument('-e', '--epochs', default=100, type=int, help='number of epochs (default: 100)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-o', '--optimizer', default='adam', type=str, metavar='OPT', help='optimizer (default: adam)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-d', '--device', default='cuda', type=str, help='device to use (default: cuda)')
parser.add_argument('-s', '--scheduler', default='warmup_cosine', type=str, help='scheduler (default: warmup_cosine)')
parser.add_argument('-t', '--topk', default=(1, ), type=tuple, help='topk acc (default: (1, ))')

best_acc1 = 0

optimizer_dict = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

scheduler_dict = {
    'warmup_linear': WarmupLinearSchedule,
    'warmup_cosine': WarmupCosineSchedule,
}

def main():
    args = parser.parse_args()
    model_config = config_dict[args.arch]
    
    device = args.device
    transform = get_transform(model_config['img_size'])
    train_loader, val_loader = get_data_loader(args.data_path, transform, args.batch_size, shuffle=True)
    
    model = build_model(DeiT, model_config).to(device)
    optimizer = optimizer_dict[args.optimizer.lower()](model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = scheduler_dict[args.scheduler](optimizer, args.epochs//5, args.epochs)
    trainer = Trainer(model, {'train':train_loader, 'validation':val_loader}, criterion, optimizer, scheduler, args.epochs, args.topk, device)
    trainer.train()
        

if __name__ == '__main__':
    main()
