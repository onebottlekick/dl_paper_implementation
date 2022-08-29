import os

from imageio import imsave, imread
import numpy as np
import torch

from common.trainer.base_trainer import BaseTrainer


class RefSR_Trainer(BaseTrainer):
    def __init__(self, args, logger, dataloader, model, criterion, optimizer):
        super().__init__(args, logger, dataloader, model, criterion, optimizer)
        
    def train(self, cur_epoch=0):
        pass
    
    def eval(self, cur_epoch=0):
        pass
    
    def test(self):
        pass