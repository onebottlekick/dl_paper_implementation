import sys
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from common.dataset.dataloader import get_dataloader
from common.logger import mkExpDir
from trainer import Trainer
from configs import args
from model import EDSR


if __name__ == '__main__':
    _logger = mkExpDir(args)
    _dataloader = get_dataloader(args) if not args.test else None
    _model = EDSR(args)
    _criterion = nn.L1Loss()
    _optimizer = torch.optim.Adam
    
    trainer = Trainer(args, _logger, _dataloader, _model, _criterion, _optimizer)
    
    if args.test:
        trainer.load(model_path=args.model_path)
        trainer.test()
        
    elif args.eval:
        trainer.load(model_path=args.model_path)
        trainer.eval()
        
    else:
        if args.resume:
            trainer.load(model_path=args.model_path)
        for epoch in range(args.start_epoch, args.num_epochs+1):
            trainer.train(cur_epoch=epoch)
            if epoch%args.val_every == 0:
                trainer.eval(cur_epoch=epoch)