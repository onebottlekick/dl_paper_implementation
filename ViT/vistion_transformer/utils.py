import os

import torch


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