import os
import sys
sys.path.append('../')
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm

from configs import args
from model import VDSR
from utils import plot_featuremaps, preprocess_img

if __name__ == '__main__':
    img = preprocess_img(args.lr_path)
    _model = VDSR(args)
    _model.load_state_dict(torch.load(args.model_path))
        
    for i in tqdm(range(args.num_res_blocks)):
        for j in range(2):
            plot_featuremaps(_model, _model.residual_layer[i][j], img, 8, 8, save_name=f'fig_{i}_{"conv" if j == 0 else "relu"}.png')
