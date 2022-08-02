import os, sys
sys.path.append('../')
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummaryX import summary
from torchvision import transforms

from utils import build_model, get_patch_shape


def cos_similarity(x, y):
    return x@y/(torch.norm(x)*torch.norm(y))


def _pos_emb_similarity(pos_embedding, nrow, ncol):
    total = nrow*ncol
    pos_embedding = pos_embedding.squeeze()
    a = torch.empty(total, total)
    for i in range(total):
        b = torch.empty(total, )
        for j in range(total):
            b[j] = cos_similarity(pos_embedding[i], pos_embedding[j])
        a[i] = b
    
    return a.view(nrow, ncol, nrow, ncol).cpu().detach().numpy()
    

def plot_pos_emb_similarity(pos_emb, nrow=7, ncol=7):
    pos_emb_similarity = _pos_emb_similarity(pos_emb, nrow, ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 10))
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]
            ax.axis('off')
            im = ax.imshow(pos_emb_similarity[i, j])
    fig.colorbar(im, ax=axes, shrink=0.8, label='Cosine similarity')
    

def plot_attention_map(img, model, img_size, device='cuda', img_channels=3):
    if isinstance(img, Image.Image):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
        ])
        x = transform(img)

    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = img.transpose(2, 0, 1)
        img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToPILImage()
        ])(torch.from_numpy(img))
        x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
        ])(img)

    elif isinstance(img, torch.Tensor):
        if img.ndim == 2:
            img = img.unsqueeze(0)
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
        ])(img)
        # img = transforms.ToPILImage()(img)
        x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
        ])(img)

    else:
        raise ValueError('img must be a PIL.Image, numpy.ndarray or torch.Tensor')


    model = model.to(device)

    _, attentions = model(x.unsqueeze(0).to(device))
    attentions = torch.stack(attentions).squeeze(1)
    attentions = attentions.cpu().detach()
    attentions = torch.mean(attentions, dim=1)
    attentions = torch.mean(attentions, dim=0).unsqueeze(0)

    residual_attentions = torch.eye(attentions.shape[1])
    augmented_attentions = attentions + residual_attentions
    augmented_attentions = augmented_attentions/augmented_attentions.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(augmented_attentions.shape)
    joint_attentions[0] = augmented_attentions[0]

    for n in range(1, augmented_attentions.shape[0]):
        joint_attentions[n] = torch.matmul(augmented_attentions[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(augmented_attentions.shape[-1]))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask/mask.max(), img.size)[..., np.newaxis]
    if img.mode == 'L':
        mask = mask.reshape(*img.size)
    result = (mask*img).astype('uint8')

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))

    ax1.set_title('Image')
    ax2.set_title('Attention Mask')
    ax3.set_title('Attention Map')

    _ = ax1.imshow(img)
    ax1.axis('off')
    _ = ax2.imshow(mask.squeeze())
    ax2.axis('off')
    _ = ax3.imshow(result)
    ax3.axis('off')
    
    
def plot_rgb_filters(filters, img_channels, nrow=4, ncol=7):
    total_len = nrow*ncol
    token_dim, num_patches = get_patch_shape((3, 32, 32), [7, 3, 3], [4, 2, 2], [2, 1, 1])
    patch_size = int(np.sqrt(token_dim/3))
    for i in range(total_len):
        plt.subplot(nrow, ncol, i%total_len + 1)
        plt.imshow(filters[i].view(img_channels, patch_size, patch_size).permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')


def model_summary(model, num_channels=1, img_size=28, device='cuda'):
    x = torch.randn(1, num_channels, img_size, img_size).to(device)
    summary(model.to(device), x)
    

def load_model(model, model_path, model_config):
    model = build_model(model, model_config)
    model.load_state_dict(torch.load(model_path))    
    model.eval()
    return model