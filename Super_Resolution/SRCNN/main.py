import sys
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from common.dataset.dataloader import get_dataloader
from common.logger import mkExpDir
from common.trainer.sisr_trainer import SISR_Trainer
from configs import args
from model import SRCNN


if __name__ == '__main__':
    _logger = mkExpDir(args)
    _dataloader = get_dataloader(args) if not args.test else None
    _model = SRCNN(args)
    _criterion = nn.MSELoss()
    _optimizer = torch.optim.SGD
    
    trainer = SISR_Trainer(args, _logger, _dataloader, _model, _criterion, _optimizer)
    trainer.params = [
        {'params': _model.patch_extraction.parameters(), 'lr': args.learning_rate},
        {'params': _model.non_linear_mapping.parameters(), 'lr': args.learning_rate},
        {'params': _model.reconstruction.parameters(), 'lr': args.learning_rate*0.1}
    ]
    
    if args.test:
        trainer.load(model_path=args.model_path)
        trainer.test()
        
    elif args.eval:
        trainer.load(model_path=args.model_path)
        trainer.eval()
        
    else:
        for epoch in range(1, args.num_epochs+1):
            trainer.train(cur_epoch=epoch)
            if epoch%args.val_every == 0:
                trainer.eval(cur_epoch=epoch)