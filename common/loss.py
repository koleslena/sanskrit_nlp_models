import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (Batch, Labels, Seq)
        # targets: (Batch, Seq)
        
        # 1. Считаем CE с reduction='none', чтобы получить ошибку по каждому слову отдельно
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # 2. Считаем уверенность модели pt
        pt = torch.exp(-ce_loss)
        
        # 3. Применяем формулу Focal Loss: (1-pt)^gamma * CE
        # Вес (1-pt)^gamma будет близок к 0 для простых слов и к 1 для сложных
        f_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return f_loss.mean()