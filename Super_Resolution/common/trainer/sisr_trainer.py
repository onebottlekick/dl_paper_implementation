import os
from PIL import Image

import numpy as np
from imageio import imsave
import torch
from torchvision import transforms

from .base_trainer import BaseTrainer
from ..utils import calc_psnr_and_ssim


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
            model_name = self.args.save_dir.strip('/') + '/model/' + self.args.log_file_name.strip('.log') + '_' + str(cur_epoch).zfill(5) + '.pth'
            torch.save(model_state_dict, model_name)
            
    
    def eval(self, cur_epoch=0):
        self.logger.info(f'Epoch {cur_epoch} evaluation')
        self.model.eval()
        with torch.no_grad():
            psnr, ssim, cnt = 0., 0., 0
            for i_batch, sample_batched in enumerate(self.dataloader['test']):
                cnt += 1
                sample_batched = self.to_device(sample_batched)
                lr = sample_batched['LR']
                hr = sample_batched['HR']
                
                sr = self.model(lr)
                
                if self.args.eval_save_results:
                    sr_save = (sr + 0.5)*255.
                    sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                    imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5) + '.png'), sr_save)
                
                _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                
                psnr += _psnr
                ssim += _ssim
            
            avg_psnr = psnr/cnt
            avg_ssim = ssim/cnt
            self.logger.info(f'PSNR (now): {avg_psnr:.3f} \t SSIM (now): {avg_ssim:.3f}')
            if avg_psnr > self.max_psnr:
                self.max_psnr = avg_psnr
                self.max_psnr_epoch = cur_epoch
            if avg_ssim > self.max_ssim:
                self.max_ssim = avg_ssim
                self.max_ssim_epoch = cur_epoch
            self.logger.info(f'PSNR (max): {self.max_psnr:.3f} on epoch: {self.max_psnr_epoch} \t SSIM (max): {self.max_ssim:.3f} on epoch: {self.max_ssim_epoch}')
                
                            
        self.logger.info('Eval Done')
    
    def test(self):
        self.logger.info('Test')
        self.logger.info(f'LR path {self.args.lr_path}')
        
        lr = Image.open(self.args.lr_path).convert('RGB')
        lr = transforms.Compose([
            # transforms.CenterCrop(self.args.img_size),
            # transforms.RandomCrop(self.args.img_size), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5 for _ in range(self.args.img_channels)], std=[0.5 for _ in range(self.args.img_channels)])
            ])(lr)
        lr = lr.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            sr = self.model(lr)
            sr = transforms.ToPILImage()((sr.squeeze(0) + 1.)*127.5)
            save_path = os.path.join(self.args.save_dir, 'save_results', f'{os.path.basename(self.args.lr_path).split(".")[0]}_{self.args.log_file_name.split(".")[0]}.png')
            sr.save(save_path)
            self.logger.info(f'output path: {save_path}')
            
        self.logger.info('Test Done')