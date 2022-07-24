import math
import os

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def pair(x):
    return x if isinstance(x, tuple) else (x, x)


def build_model(model, config):
    model = model(**config)
    return model


def save_checkpoint(state, architecture, path='./'):
    torch.save(state, os.path.join(path, f'{architecture}.pth.tar'))
    

def lr_decay(optimizer, epoch, lr):
    lr = lr*(0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.shape[0]
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0/batch_size))
        
        return results
    
    
def get_transform(img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(pair(img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return transform
    
    
def get_data_loader(data_path, transform, batch_size, shuffle, split_ratio=0.7):
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split
    
    dataset = ImageFolder(data_path, transform=transform)
    
    num_train = int(len(dataset)*split_ratio)
    num_val = len(dataset)-num_train    
    train_data, val_data = random_split(dataset, [num_train, num_val])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, val_loader


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
    
    
class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class Trainer:
    def __init__(self, model, dataloader_dict, criterion, optimizer, scheduler, num_epochs, topk, model_path, device):
        self.model = model.to(device)
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.dataloader_dict = dataloader_dict
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.topk = topk
        self.device = device
        self.running_loss, self.running_acc = {}, {}
        self.model_path = model_path
    
    # TODO calc topk acc
    def train(self):
        for epoch in range(self.num_epochs):
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                self.running_loss[phase], self.running_acc[phase] = 0.0, 0.0

                with tqdm(self.dataloader_dict[phase], unit='Batch') as t:
                    t.set_description(f'{phase} Epoch: {epoch+1}')
                    for inputs, targets in t:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs, _ = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                            
                            if loss.item() < self.best_loss and not self.model.training:
                                torch.save(self.model.state_dict(), self.model_path)
                                self.best_loss = loss.item()

                            t.set_postfix(loss=f'{loss.item():.6f}')

                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                    if phase == 'train':
                        self.scheduler.step()
                    else:
                        print()
                        print()
