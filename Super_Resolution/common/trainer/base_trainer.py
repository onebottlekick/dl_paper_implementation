from abc import *
import os

from imageio import imsave, imread
import numpy as np
import torch


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, logger, dataloader, model, criterion, optimizer):
        self.args = args
        self.logger = logger
        self.device = 'cpu' if self.args.cpu else 'cuda'
        self.dataloader = dataloader
        self.model = model.to(self.device)
        self.criterion = criterion
        
        self.params = model.parameters()
        self.optimizer = optimizer(self.params, lr=args.learning_rate)
        
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0
                
    def load(self, model_path=None):
        if model_path:
            self.logger.info('load_model_path' + model_path)
            model_state_dict_save = {k:v for k, v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)
            
    def to_device(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        
        return sample_batched
    
    @abstractmethod
    def train(self, cur_epoch=0):
        pass
    
    @abstractmethod
    def eval(self, cur_epoch=0):
        pass
    
    @abstractmethod
    def test(self):
        pass
