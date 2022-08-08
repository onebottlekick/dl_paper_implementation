import os
from PIL import Image

import matplotlib.pyplot as plt
import torch
import numpy as np


def preprocess_img(path):
    img = Image.open(os.path.join('../', path)).convert('RGB')
    img = np.array(img)
    img = img/127.5 - 1.
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    return img


def plot_featuremaps(model, features, img, n_rows, n_cols, save_path='results', save_name='result.png'):
    result = LayerActivations(features)
    model(img)
    activations = result.features

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

    for row in range(n_rows):
        for column in range(n_cols):
            axis = axes[row][column]
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            axis.imshow(activations[0][row*n_cols+column], cmap='gray')
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, save_name))
    else:
        plt.show()
    

class LayerActivations:
    features = []
    def __init__(self, model_feature, layer_num=None):
        if layer_num is not None:
            self.hook = model_feature[layer_num].register_forward_hook(self.hook_fn)
        else:
            self.hook = model_feature.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output.detach().numpy()
        
    def remove(self):
        self.hook.remove()