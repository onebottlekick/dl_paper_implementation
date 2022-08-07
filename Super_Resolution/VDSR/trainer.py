import sys
sys.path.append('../')

import torch
import torch.nn as nn

from common.trainer.sisr_trainer import SISR_Trainer


def lr_decay(args, cur_epoch):
    lr = args.learning_rate*(args.lr_decay_rate**(cur_epoch//args.lr_decay_step))
    
    return lr


class Trainer(SISR_Trainer):
    def __init__(self, args, logger, dataloader, model, criterion, optimizer):
        super().__init__(args, logger, dataloader, model, criterion, optimizer)
        
        self.optimizer = optimizer(model.parameters(), momentum=args.momentum, weight_decay=args.weight_decay)
        
    def train(self, cur_epoch=0):
        self.model.train()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_decay(self.args, cur_epoch)
        self.logger.info(f'Current epoch learning rate: {self.optimizer.param_groups[0]["lr"]}')
        
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()
            
            sample_batched = self.to_device(sample_batched)
            lr = sample_batched['LR']
            hr = sample_batched['HR']
            
            sr = self.model(lr)
            
            is_print = ((i_batch + 1) %self.args.print_every == 0)
            
            loss = self.criterion(sr, hr)
            if (is_print):
                self.logger.info(f'Epoch: {cur_epoch}\tbatch: {i_batch + 1}')
                self.logger.info(f'loss: {loss.item():.4f}')
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            
        if cur_epoch%self.args.save_every == 0:
            self.logger.info('saving model...')
            model_state_dict = self.model.state_dict()
            model_name = self.args.save_dir.strip('/') + '/model/' + self.args.log_file_name.strip('.log') + '_' + str(cur_epoch).zfill(5) + '.pth'
            torch.save(model_state_dict, model_name)