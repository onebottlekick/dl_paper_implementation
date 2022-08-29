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
from model import Generator
from loss import VGGLoss, AdversarialLoss


if __name__ == '__main__':
    _logger = mkExpDir(args)
    _dataloader = get_dataloader(args) if not args.test else None
    _model = Generator(args)
    _criterion = nn.MSELoss() if 'mse'.lower() in args.loss.lower() else VGGLoss(args)
    _optimizer = torch.optim.Adam
    _adversarial_loss = AdversarialLoss(args) if 'perceptual'.lower() in args.loss.lower() else None
    
    trainer = Trainer(args, _logger, _dataloader, _model, _criterion, _optimizer, adversarial_loss=_adversarial_loss)
    
    if args.test:
        trainer.load(model_path=args.model_path)
        trainer.test()
        
    elif args.eval:
        trainer.load(model_path=args.model_path)
        trainer.eval()
        
    else:
        if args.resume or _adversarial_loss:
            trainer.load(model_path=args.model_path)
            _adversarial_loss.discriminator.load_state_dict(torch.load(args.discriminator_path))

        for epoch in range(args.start_epoch, args.num_epochs+1):
            trainer.train(cur_epoch=epoch)
            if epoch%args.val_every == 0:
                trainer.eval(cur_epoch=epoch)