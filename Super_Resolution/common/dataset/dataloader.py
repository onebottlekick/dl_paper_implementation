from importlib import import_module

from torch.utils.data import DataLoader


def get_sisr_dataloader(args):
    m = import_module('common.' + 'dataset.' + 'sisr_dataset')
    
    train_data = getattr(m, 'SISR_Dataset')(args, args.train_dataset_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    test_data = getattr(m, 'SISR_Dataset')(args, args.test_dataset_dir)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    dataloader = {'train': train_loader, 'test': test_loader}
    
    return dataloader


def get_refsr_dataloader(args):
    m = import_module('common.' + 'dataset.' + 'refsr_dataset')
    
    train_data = getattr(m, 'RefSR_Dataset')(args, args.train_dataset_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    test_data = getattr(m, 'RefSR_Dataset')(args, args.test_dataset_dir)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    dataloader = {'train': train_loader, 'test': test_loader}
    
    return dataloader
