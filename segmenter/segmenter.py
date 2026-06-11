import torch
import torch.nn as nn

from segmenter.decoder import PointerGeneratorDecoder
from segmenter.encoder import BiLSTMEncoder

class SanskritPointerSegmenter(nn.Module):
    def __init__(self, vocab_size, emb_dim, device, hidden_dim=512, n_layers=6, n_layers_dec=2, dropout=0.2, all_bi=False):
        super().__init__()
        self.encoder = BiLSTMEncoder(vocab_size, emb_dim, hidden_dim, n_layers, dropout)
        self.decoder = PointerGeneratorDecoder(vocab_size, emb_dim, hidden_dim, n_layers_dec, dropout)
        self.device = device
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_layers_dec = n_layers_dec
        self.hidden_dim = hidden_dim
        self.all_bi = all_bi

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

        # логика инициализации декодера для ГЛУБОКОГО BiLSTM (all_bi == True)
        half = self.hidden_dim // 2

        # TODO
        if self.device.type == 'mps':
            # 2. Извлекаем состояние последнего реального символа для каждого слова в батче
            # Мы просто берем из каждого примера в батче строку с индексом (длина - 1)
            last_char_states = []
            for i in range(batch_size):
                # Индекс последней буквы для i-го примера
                last_idx = lengths[i] - 1
                if self.all_bi:
                    # Забираем зрелый финал прямого прохода (с конца строки)
                    fw_state = encoder_outputs[i, last_idx, :half]
                    # Забираем зрелый финал обратного прохода (с самого начала строки, индекс 0)
                    bw_state = encoder_outputs[i, 0, half:]
                    
                    # Склеиваем их в один полноценный контекстный вектор [hidden_dim]
                    combined = torch.cat([fw_state, bw_state], dim=-1)
                    last_char_states.append(combined)
                else:
                    # Вытаскиваем вектор [hidden_dim] и добавляем его в список
                    last_char_states.append(encoder_outputs[i, last_idx, :])

            # Собираем список векторов в один тензор формы [1, batch, hidden_dim]
            last_hidden = torch.stack(last_char_states).unsqueeze(0).contiguous()

            # Инициализируем hidden_states
            hidden_states = []
            for _ in range(self.n_layers_dec):
                h = last_hidden.clone()
                # Создаем c сразу на MPS устройстве
                c = torch.zeros(1, batch_size, self.hidden_dim, device=self.device, dtype=last_hidden.dtype)
                hidden_states.append((h, c))
        else:
            if self.all_bi:
                # 2. Извлекаем скрытое состояние последнего реального символа
                # Вырезаем индексы для прямого прохода
                last_idx_fw = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, half).to(self.device)
                fw_hidden = encoder_outputs[:, :, :half].gather(1, last_idx_fw) # [batch, 1, half]
                
                # Обратный проход всегда завершается в нулевом индексе
                bw_hidden = encoder_outputs[:, 0, half:].unsqueeze(1) # [batch, 1, half]
                
                # Соединяем их вместе и превращаем в форму [1, batch, hidden_dim]
                last_hidden = torch.cat([fw_hidden, bw_hidden], dim=-1).transpose(0, 1)
            else:
                # 2. Извлекаем скрытое состояние последнего реального символа
                # Сдвигаем длины на -1 для получения индексов (0-based)
                last_idx = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, self.hidden_dim).to(self.device)
                
                # Выбираем нужные векторы из encoder_outputs
                # last_hidden будет иметь форму [1, batch, hidden_dim]
                last_hidden = encoder_outputs.gather(1, last_idx).transpose(0, 1)

            # Инициализируем декодер этим состоянием
            hidden_states = [
                (
                    last_hidden.clone(), 
                    torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
                )
                for _ in range(self.n_layers_dec)
            ]
        
        # 3. Первый символ для декодера всегда <SOS> 
        input_char = trg[:, 0].unsqueeze(1)

        encoder_projected = self.decoder.attention.U(encoder_outputs)
        
        for t in range(trg_len):
            # Декодер возвращает логарифм распределения (уже с учетом Pointer)
            log_prob, hidden_states = self.decoder(
                input_char, 
                hidden_states, 
                encoder_outputs,  
                src, # передаем src для механизма копирования
                src_mask, # передаем маску на src
                encoder_projected=encoder_projected
            )
            
            outputs[:, t] = log_prob
            
            # Решаем: использовать правильный символ или предсказание модели
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = log_prob.argmax(1)
            if t < trg_len - 1:
                input_char = trg[:, t + 1].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs