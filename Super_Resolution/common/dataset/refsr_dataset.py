import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class RefSR_Dataset(Dataset):
    def __init__(self, args, input_path, ref_path):
        self.args = args
        
        self.input_list = sorted([os.path.join(input_path, name) for name in os.listdir(input_path)])
        self.ref_list = sorted([os.path.join(ref_path, name) for name in os.listdir(ref_path)])
        
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(args.img_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
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
        
        Ref = Image.open(self.ref_list[idx]).convert('RGB')
        Ref = self.hr_transform(Ref)    
        
        sample = {'HR': HR,
                  'LR': LR,
                  'Ref': Ref}
        
        return sample