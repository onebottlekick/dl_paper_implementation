import argparse

import torch
import torch.nn as nn

from configs import config_dict
from loss import DistillationLoss
from model import DeiT
from teacher_model import ResNet50
from utils import Trainer, WarmupCosineSchedule, WarmupLinearSchedule, build_model, get_transform, get_data_loader, build_teacher_model

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
parser.add_argument('--model-path', default='./experiments/pretrained/model.pth', type=str, help='path to model')
parser.add_argument('--monitor', default='loss', type=str, help='monitor (default: loss)')

parser.add_argument('--teacher-model', default='resnet', type=str, help='teacher model (default: resnet)')
parser.add_argument('--teacher-type', default='resnet', type=str, help='teacher model type (default: resnet)')
parser.add_argument('--distillation', action='store_ture', help='distillation (default: True)')
parser.add_argument('--distil-type', default='hard', type=str, help='distil_type (default: hard)')
parser.add_argument('--distil-token-type', default='cls+distil', type=str, help='distil_token_type (default: cls+distil)')
parser.add_argument('--alpha', default=0.5, type=float, help='distillation alpha (default: 0.5)')
parser.add_argument('--tau', default=3.0, type=float, help='distillation tau (default: 3.0)')

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
    teacher_model = build_teacher_model(ResNet50, 1000)
    optimizer = optimizer_dict[args.optimizer.lower()](model.parameters(), lr=args.lr)
    base_criterion = nn.CrossEntropyLoss()
    criterion = DistillationLoss(base_criterion, teacher_model, args.distil_type, args.alpha, args.tau, args.teacher_type)
    scheduler = scheduler_dict[args.scheduler](optimizer, args.epochs//5, args.epochs)
    trainer = Trainer(model, {'train':train_loader, 'validation':val_loader}, criterion, optimizer, scheduler, args.epochs, args.topk, args.model_path, device, teacher_model, args.monitor, args.distillation, args.distil_token_type)
    trainer.train()
        

if __name__ == '__main__':
    main()
