import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, args, input_path, ref_path=None):
        self.args = args
        self.use_ref = False
        
        self.input_list = sorted([os.path.join(input_path, name) for name in os.listdir(input_path)])
        if ref_path:
            self.use_ref = True
            self.ref_list = sorted([os.path.join(ref_path, name) for name in os.listdir(ref_path)])
        
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5 for _ in range(args.img_channels)], std=[0.5 for _ in range(args.img_channels)])
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_size[0]//args.lr_scale, args.img_size[1]//args.lr_scale), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
                
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        HR = Image.open(self.input_list[idx]).convert('RGB')
        HR = self.hr_transform(HR)
        
        LR = self.lr_transform(HR)
        
        sample = {'HR': HR,
                  'LR': LR}
        
        if self.use_ref:
            Ref = Image.open(self.ref_list[idx]).convert('RGB')
            Ref = self.hr_transform(Ref)
            
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
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5 for _ in range(args.img_channels)], std=[0.5 for _ in range(args.img_channels)])
        ])
                
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        _HR = Image.open(self.input_list[idx]).convert('RGB')
        w, h = _HR.size
        HR = self.transform(_HR)
        
        LR = _HR.resize((w//self.args.lr_scale, h//self.args.lr_scale), resample=Image.BICUBIC)
        LR = self.transform(LR)
        
        sample = {'HR': HR,
                  'LR': LR}
        
        if self.use_ref:
            Ref = Image.open(self.ref_list[idx]).convert('RGB')
            Ref = self.transform(Ref)
            
            sample.update({'Ref': Ref})
        
        
        return sample