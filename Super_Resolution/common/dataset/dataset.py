import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ToTensor:
    def __call__(self, x):
        return torch.tensor(x)
    
    
class Identity:
    def __call__(self, x):
        return x


class TrainDataset(Dataset):
    def __init__(self, args, input_path, ref_path=None):
        self.args = args
        self.use_ref = False
        
        self.input_list = sorted([os.path.join(input_path, name) for name in os.listdir(input_path)])
        if ref_path:
            self.use_ref = True
            self.ref_list = sorted([os.path.join(ref_path, name) for name in os.listdir(ref_path)])
        
        self.hr_transform = transforms.Compose([
            ToTensor(),
            # transforms.RandomCrop(args.img_size),
            transforms.CenterCrop(args.img_size), # if self.args.batch_size != 1 else Identity(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        HR = Image.open(self.input_list[idx]).convert('RGB')
        HR = np.array(HR).transpose(2, 0, 1)
        HR = HR/127.5 - 1.
        HR = self.hr_transform(HR).float()
        c, h, w = HR.shape
        
        LR = transforms.Resize((h//self.args.lr_scale, w//self.args.lr_scale), interpolation=transforms.InterpolationMode.BICUBIC)(HR)
        
        sample = {'HR': HR,
                  'LR': LR}
        
        if self.use_ref:
            Ref = Image.open(self.ref_list[idx]).convert('RGB')
            Ref = np.array(Ref).transpose(2, 0, 1)
            Ref = Ref/127.5 - 1.
            c, h, w = Ref.shape
            Ref = self.hr_transform(Ref).float()
            
            sample.update({'Ref': Ref})
        
        
        return sample
    
    
class TestDataset(Dataset):
    def __init__(self, args, input_path, ref_path=None):
        self.args = args
        self.use_ref = False
        
        self.input_list = sorted([os.path.join(input_path, name) for name in os.listdir(input_path)])
        if ref_path:
            self.use_ref = True
            self.ref_list = sorted([os.path.join(ref_path, name) for name in os.listdir(ref_path)])
                
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        HR = Image.open(self.input_list[idx]).convert('RGB')
        HR = np.array(HR).transpose(2, 0, 1)
        HR = HR/127.5 - 1.
        HR = ToTensor()(HR).float()
        c, h, w = HR.shape
        
        LR = transforms.Resize((h//self.args.lr_scale, w//self.args.lr_scale), interpolation=transforms.InterpolationMode.BICUBIC)(HR)
        
        sample = {'HR': HR,
                  'LR': LR}
        
        if self.use_ref:
            Ref = Image.open(self.ref_list[idx]).convert('RGB')
            Ref = np.array(Ref).transpose(2, 0, 1)
            Ref = Ref/127.5 - 1.
            Ref = ToTensor()(Ref).float()
            
            sample.update({'Ref': Ref})
        
        
        return sample