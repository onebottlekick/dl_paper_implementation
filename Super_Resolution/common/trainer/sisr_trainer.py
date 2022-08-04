import os
from PIL import Image

import numpy as np
from imageio import imsave
import torch
from torchvision import transforms

from .base_trainer import BaseTrainer


class SISR_Trainer(BaseTrainer):
    def __init__(self, args, logger, dataloader, model, criterion, optimizer):
        super().__init__(args, logger, dataloader, model, criterion, optimizer)
        
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
            if (is_print):
                self.logger.info(f'Epoch: {cur_epoch}\tbatch: {i_batch + 1}')
                self.logger.info(f'loss: {loss.item():.4f}')
            loss.backward()
            self.optimizer.step()
            
        if cur_epoch%self.args.save_every == 0:
            self.logger.info('saving model...')
            model_state_dict = self.model.state_dict()
            model_name = self.args.save_dir.strip('/') + '/model/model_' + str(cur_epoch).zfill(5) + '.pth'
            torch.save(model_state_dict, model_name)
            
    
    def eval(self, cur_epoch=0):
        self.logger.info(f'Epoch {cur_epoch} evaluation')
        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader['test']):
                sample_batched = self.to_device(sample_batched)
                lr = sample_batched['LR']
                hr = sample_batched['HR']
                
                sr = self.model(lr)
                
                if self.args.eval_save_results:
                    sr_save = (sr + 0.5)*255.
                    sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                    imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5) + '.png'), sr_save)
        self.logger.info('Eval Done')
    
    def test(self):
        self.logger.info('Test')
        self.logger.info(f'LR path {self.args.lr_path}')
        
        lr = Image.open(self.args.lr_path).convert('RGB')
        lr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5 for _ in range(self.args.img_channels)], std=[0.5 for _ in range(self.args.img_channels)])])(lr)
        lr = lr.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            sr = self.model(lr)
            sr = transforms.ToPILImage()((sr.squeeze(0)*0.5) + 0.5)
            save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            sr.save(save_path)
            self.logger.info(f'output path: {save_path}')
            
        self.logger.info('Test Done')