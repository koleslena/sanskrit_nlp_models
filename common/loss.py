import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=-100, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)
        
        # Передаем веса классов в CE, если они есть
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                  ignore_index=self.ignore_index, weight=self.weight)
        
        pt = torch.exp(-ce_loss)
        f_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Считаем маску реальных (не паддинг) токенов
        mask = (targets != self.ignore_index).float()
        
        # Усредняем только по реальным словам
        return f_loss.sum() / mask.sum().clamp(min=1)


class SegmenterFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, log_probs, targets):
        """
        log_probs: [batch_size * seq_len, vocab_size] - логарифмы вероятностей из модели
        targets: [batch_size * seq_len] - истинные индексы символов целевой строки
        """
        # 1. Создаем маску для игнорирования <PAD> токенов
        valid_mask = (targets != self.ignore_index)
        
        # 2. Вытаскиваем log(p_t) для каждого правильного класса
        # Из каждой строки тензора log_probs берем значение по индексу из targets
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 3. Восстанавливаем чистую вероятность p_t для расчета динамического веса
        # Ограничиваем сверху 1.0 на случай микроошибок плавающей точки
        pt = torch.exp(log_pt).clamp(max=1.0)
        
        # 4. Вычисляем коэффициент Focal Loss
        # Чем выше уверенность pt, тем ближе коэффициент к нулю
        focal_weight = self.alpha * ((1 - pt) ** self.gamma)
        
        # 5. Итоговый лосс для каждого токена
        loss = -focal_weight * log_pt
        
        # 6. Обнуляем лосс там, где были паддинги, и усредняем по реальным символам
        loss = loss * valid_mask.float()
        
        return loss.sum() / valid_mask.sum()