import os
import sys
sys.path.append('../')
import warnings
warnings.filterwarnings('ignore')

import torch

from configs import args
from model import SRCNN
from utils import plot_featuremaps, preprocess_img

if __name__ == '__main__':
    img = preprocess_img(args.lr_path)
    _model = SRCNN(args)
    _model.load_state_dict(torch.load(args.model_path))
    
    plot_featuremaps(_model, _model.patch_extraction, img, args.n_rows1, args.n_cols1, save_name=args.first_layer_fig_name)
    plot_featuremaps(_model, _model.non_linear_mapping, img, args.n_rows2, args.n_cols2, save_name=args.second_layer_fig_name)