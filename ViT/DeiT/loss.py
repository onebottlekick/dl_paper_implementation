import torch
import torch.nn as nn


class DistillationLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, distil_type, alpha, tau):
        super().__init__()
        
        self.criterion = base_criterion
        self.teacher_model = teacher_model
        self.distil_type = distil_type
        self.alpha = alpha
        self.tau = tau
        
    def forward(self, inputs, cls_tokens, distil_tokens, labels):
        base_loss = self.criterion(cls_tokens, labels)
        if self.distil_type == 'none':
            return base_loss
        
        with torch.no_grad():
            teacher_outputs, _, _ = self.teacher_model(inputs)
            
        if self.distil_type == 'soft':
            distil_loss = nn.KLDivLoss(reduction='sum', log_target=True)(nn.LogSoftmax(dim=1)(distil_tokens/self.tau), nn.LogSoftmax(dim=1)(teacher_outputs/self.tau))*(self.tau*self.tau)/distil_tokens.numel()
        elif self.distil_type == 'hard':
            distil_loss = nn.CrossEntropyLoss()(distil_tokens, teacher_outputs.argmax(dim=1))
        
        loss = base_loss*(1 - self.alpha) + distil_loss*self.alpha
        
        return loss