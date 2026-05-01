import torch
import torch.nn as nn

from segmenter.decoder import PointerGeneratorDecoder
from segmenter.encoder import BiLSTMEncoder

class SanskritPointerSegmenter(nn.Module):
    def __init__(self, vocab_size, emb_dim, device, hidden_dim=512, n_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = BiLSTMEncoder(vocab_size, emb_dim, hidden_dim, n_layers, dropout)
        self.decoder = PointerGeneratorDecoder(vocab_size, emb_dim, hidden_dim, n_layers, dropout)
        self.device = device
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len] - вход в SLP1 (без пробелов)
        # trg: [batch, trg_len] - выход (с разрывами и изменениями)
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        
        # Сюда будем собирать логарифмы вероятностей
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        
        # 1. Энкодер создает контекст
        encoder_outputs, lengths = self.encoder(src)

        # Создаем маску для внимания
        # True (1) там где есть символ, False (0) там где PAD
        src_mask = (src != 0)

        # TODO
        if self.device.type == 'mps':
            # 2. Извлекаем состояние последнего реального символа для каждого слова в батче
            # Мы просто берем из каждого примера в батче строку с индексом (длина - 1)
            last_char_states = []
            for i in range(batch_size):
                # Индекс последней буквы для i-го примера
                last_idx = lengths[i] - 1
                # Вытаскиваем вектор [hidden_dim] и добавляем его в список
                last_char_states.append(encoder_outputs[i, last_idx, :])

            # Собираем список векторов в один тензор формы [1, batch, hidden_dim]
            last_hidden = torch.stack(last_char_states).unsqueeze(0).contiguous()

            # Инициализируем hidden_states
            hidden_states = []
            for _ in range(self.n_layers):
                h = last_hidden.clone()
                # Создаем c сразу на MPS устройстве
                c = torch.zeros(1, batch_size, self.hidden_dim, device=self.device, dtype=last_hidden.dtype)
                hidden_states.append((h, c))
        else:
            # 2. Извлекаем скрытое состояние последнего реального символа
            # Сдвигаем длины на -1 для получения индексов (0-based)
            last_idx = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, self.hidden_dim)
            
            # Выбираем нужные векторы из encoder_outputs
            # last_hidden будет иметь форму [1, batch, hidden_dim]
            last_hidden = encoder_outputs.gather(1, last_idx).transpose(0, 1)

            # Инициализируем декодер этим состоянием
            hidden_states = [
                (
                    last_hidden.clone(), 
                    torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
                )
                for _ in range(self.n_layers)
            ]
        
        # 3. Первый символ для декодера всегда <SOS> 
        input_char = trg[:, 0].unsqueeze(1)
        
        for t in range(trg_len):
            # Декодер возвращает логарифм распределения (уже с учетом Pointer)
            log_prob, hidden_states = self.decoder(
                input_char, 
                hidden_states, 
                encoder_outputs,  
                src, # передаем src для механизма копирования
                src_mask  # передаем маску на src
            )
            
            outputs[:, t] = log_prob
            
            # Решаем: использовать правильный символ или предсказание модели
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = log_prob.argmax(1)
            if t < trg_len - 1:
                input_char = trg[:, t + 1].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs