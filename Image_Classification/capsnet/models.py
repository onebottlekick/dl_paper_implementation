from pyrsistent import s
import torch
import torch.nn as nn


class Squashing(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, s):
        s2 = (s**2).sum(dim=-1, keepdim=True)
        
        return (s2/(1 + s2))*(s/torch.sqrt(s2 + self.epsilon))
    
    
class Router(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_dim, out_dim, iterations):
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.iterations = iterations
        self.softmax = nn.softmax(dim=1)
        self.squashing = Squashing()
        self.weight = nn.Parameter(torch.randn(in_capsules, out_capsules, in_dim, out_dim), requires_grad=True)
        
    def forward(self, u):
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        b = u.new_zeros(u.shape[0], self.in_capsules, self.out_capsules)
        for i in range(self.iterations):
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squashing(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a
            
        return v
    
    
class MarginLoss(nn.Module):
    def __init__(self, n_labels, m_positive, m_negative, lambda_=0.5):
        super().__init__()
        self.n_labels = n_labels
        self.lambda_ = lambda_
        self.m_positive = m_positive
        self.m_negative = m_negative
        
    def forward(self, v, labels):
        v_norm = torch.sqrt((v**2).sum(dim=-1))
        labels = torch.eye(self.n_labels, device=labels.device)[labels]
        loss = labels*nn.ReLU()(self.m_positive - v_norm) + self.lambda_*(1 - labels)*nn.ReLU()(v_norm - self.m_negative)
        
        return loss.sum(dim=-1).mean()
    
    
class CapsuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(256, 32*8, kernel_size=9, stride=2)
        
        self.squashing = Squashing()
        
        self.digit_caps = Router(32*6*6, 10, 8, 16, 3)
        
        self.decoder = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu((self.conv1(x)))
        x = self.conv2(x)
        caps = x.view(x.shape[0], 8, 32*6*6).permute(0, 2, 1)
        caps = self.squashing(caps)
        caps = self.digit_caps(caps)
        
        with torch.no_grad():
            pred = (caps**2).sum(dim=-1).argmax(-1)
            mask = torch.eye(10, device=x.device)[pred]
            
        reconstructions = self.decoder((caps*mask[:, :, None]).view(x.shape[0], -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)
        
        return caps, reconstructions, pred