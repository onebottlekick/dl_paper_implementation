from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from common.trainer.sisr_trainer import SISR_Trainer


class Trainer(SISR_Trainer):
    def __init__(self, args, logger, dataloader, model, criterion, optimizer, adversarial_loss=None):
        super().__init__(args, logger, dataloader, model, criterion, optimizer)
        
        self.adversarial_loss = adversarial_loss
        self.adversarial_loss_weight = args.adversarial_loss_weight
        self.optimizer = optimizer(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.epsilon)
        
    def train(self, cur_epoch=0):
        self.model.train()
        self.logger.info(f'Current epoch learning rate: {self.optimizer.param_groups[0]["lr"]}')
        
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()
            
            sample_batched = self.to_device(sample_batched)
            lr = sample_batched['LR']
            hr = sample_batched['HR']
            
            sr = self.model(lr)
            
            is_print = ((i_batch + 1) %self.args.print_every == 0)
            
            loss = self.criterion(sr, hr)
            if 'vgg' in self.args.loss.lower():
                loss = loss*((1/12.75)**2)
            
            if (is_print):
                self.logger.info(f'Epoch: {cur_epoch}\tbatch: {i_batch + 1}')
                self.logger.info(f'content_loss: {loss.item():.4f}')
            if self.adversarial_loss is not None:
                adversarial_loss = self.adversarial_loss(sr, hr)
                loss += adversarial_loss*self.adversarial_loss_weight
                if is_print:
                    self.logger.info(f'adversarial_loss: {adversarial_loss.item():.4f}')
                    self.logger.info(f'total_loss: {loss.item():.4f}')
            loss.backward()
            self.optimizer.step()
            
        if cur_epoch%self.args.save_every == 0:
            self.logger.info('saving model...')
            model_state_dict = self.model.state_dict()
            model_name = self.args.save_dir.strip('/') + '/model/' + self.args.log_file_name.strip('.log') + '_' + str(cur_epoch).zfill(5) + '.pth'
            torch.save(model_state_dict, model_name)
            discriminator_state_dict = self.adversarial_loss.discriminator.state_dict()
            torch.save(discriminator_state_dict, self.args.save_dir.strip('/') + '/model/' + 'discriminator' + '.pth')
