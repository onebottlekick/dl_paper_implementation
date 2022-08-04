import glob
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imageio import imread


class ToTensor:
    def __call__(self, sample):
        HR = sample['HR']        
        HR = HR.transpose((2,0,1))
        LR = sample['LR']
        LR = LR.transpose((2,0,1))
        LR_sr = sample['LR_sr']
        LR_sr = LR_sr.transpose((2,0,1))
        
        tensor = {'LR': torch.from_numpy(LR).float(),
                  'HR': torch.from_numpy(HR).float(),
                  'LR_sr': torch.from_numpy(LR_sr).float()}
 
        return tensor


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted([os.path.join(args.train_dataset_dir, name) for name in os.listdir(args.train_dataset_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR = imread(self.input_list[idx])
        HR = HR.astype(np.float32)
        HR = HR / 127.5 - 1.
        h,w = HR.shape[:2]

        LR = np.array(Image.fromarray(np.uint8(HR)).resize((w//4, h//4), Image.BICUBIC))
        LR = LR.astype(np.float32)
        LR = LR / 127.5 - 1.
        
        LR_sr = np.array(Image.fromarray(np.uint8(LR)).resize((w, h), Image.BICUBIC))
        LR_sr = LR_sr.astype(np.float32)
        LR_sr = LR_sr / 127.5 - 1.
        
        sample = {'HR': HR,  
                  'LR': LR,
                  'LR_sr': LR_sr}

        if self.transform:
            sample = self.transform(sample)
            
        return sample


class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(glob.glob(os.path.join(args.test_dataset_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        h, w = h//4*4, w//4*4
        HR = HR[:h, :w, :]
        HR = HR.astype(np.float32)
        HR = HR / 127.5 - 1.

        LR = np.array(Image.fromarray(np.uint8(HR)).resize((w//4, h//4), Image.BICUBIC))
        LR = LR.astype(np.float32)
        LR = LR / 127.5 - 1.
        
        LR_sr = np.array(Image.fromarray(np.uint8(LR)).resize((w, h), Image.BICUBIC))
        LR_sr = LR_sr.astype(np.float32)
        LR_sr = LR_sr / 127.5 - 1.
        
        sample = {'HR': HR,  
                  'LR': LR,
                  'LR_sr': LR_sr}

        if self.transform:
            sample = self.transform(sample)
            
        return sample
