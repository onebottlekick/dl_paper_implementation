import math
import os

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def build_model(model, config):
    model = model(**config)
    return model


def build_teacher_model(model, num_classes=1000):
    model = model(num_classes)
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
        transforms.Resize((img_size, img_size)),
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


# TODO remove hardcoded topk
class Trainer:
    def __init__(self, model, dataloader_dict, criterion, optimizer, scheduler, num_epochs, topk, model_path, device, teacher_model=None, monitor='loss', distillation=False, distil_token_type='cls+distil'):
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
        self.teacher_model = teacher_model
        self.monitor = monitor
        self.model_path = model_path
        self.distillation = distillation
        self.distil_token_type = distil_token_type
    
    # TODO 
    # - distillation
    def train(self):
        for epoch in range(self.num_epochs):
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                top1, top5 = 0.0, 0.0
                total_loss = 0.0
                with tqdm(self.dataloader_dict[phase], unit='Batch') as t:
                    t.set_description(f'{phase} Epoch: {epoch+1}')
                    for idx, (inputs, targets) in enumerate(t):
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            if self.distillation:
                                cls_tokens, distil_tokens,_ = self.model(inputs)
                                
                                if self.distil_token_type == 'cls+distil':
                                    _loss = self.criterion(inputs, cls_tokens, distil_tokens, targets)
                                elif self.distil_token_type == 'distil':
                                    _loss = self.criterion(inputs, distil_tokens, distil_tokens, targets)
                                elif self.distil_token_type == 'cls':
                                    _loss = self.criterion(inputs, cls_tokens, cls_tokens, targets)
                                else:
                                    raise ValueError(f'Unknown distil token type: {self.distil_token_type}, use ["cls", "distil", "cls+distil"]')
                                
                                total_loss += _loss.item()
                                loss = total_loss/(idx+1)/cls_tokens.shape[0]
                                _top1, _top5 = accuracy(cls_tokens, targets, topk=(1, 5))
                                top1 += _top1.item()
                                top5 += _top5.item()
                                
                            else:
                                outputs = self.model(inputs)
                                _loss = self.criterion(outputs, targets)
                                total_loss += _loss.item()
                                loss = total_loss/(idx+1)/outputs.shape[0]
                                _top1, _top5 = accuracy(outputs, targets, topk=(1, 5))
                                top1 += _top1.item()
                                top5 += _top5.item()
                                
                                

                            t.set_postfix(loss=f'{loss:.6f}', top1=f'{top1/(idx+1):.2f}%', top5=f'{top5/(idx+1):.2f}%')

                            if phase == 'train':
                                _loss.backward()
                                self.optimizer.step()
                                
                    if self.monitor == 'loss':
                        if loss < self.best_loss and not self.model.training:
                            torch.save(self.model.state_dict(), self.model_path)
                            self.best_loss = loss
                    
                    elif self.monitor == 'acc':
                        if top1/(idx+1) > self.best_acc and not self.model.training:
                            torch.save(self.model.state_dict(), self.model_path)
                            self.best_acc = top1/(idx+1)

                    if phase == 'train':
                        if self.scheduler is not None:
                            self.scheduler.step()
                    else:
                        print()
                        print()