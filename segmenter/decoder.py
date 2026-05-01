import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch, hidden_dim] (из последнего слоя декодера)
        # encoder_outputs: [batch, seq_len, hidden_dim]
        
        hidden_with_time = hidden.unsqueeze(1) # [batch, 1, hidden_dim]
        
        # Вычисляем скоры (score)
        # 1. Получаем энергию (energy) -> [batch, seq_len, hidden_dim]
        energy = torch.tanh(self.W(hidden_with_time) + self.U(encoder_outputs))
        
        # 2. Сжимаем до 1 через слой v -> [batch, seq_len, 1]
        score = self.v(energy)

        # 3. Применяем маску
        if mask is not None:
            # mask: [batch, seq_len] (True - реальный символ, False - PAD)
            # Исходная маска: [batch, seq_len].
            # Добавляем измерение, чтобы стало: [batch, seq_len, 1]
            mask = mask.unsqueeze(2) 
            # Теперь размерности идеально совпадают, PAD получаем -1e9
            score = score.masked_fill(mask == 0, -1e9)
        
        # Веса внимания (attention weights)
        # 4. Вычисляем веса. PAD токены (где было -1e9) станут строгими нулями
        attention_weights = F.softmax(score, dim=1) # [batch, seq_len, 1]
        
        # 5. Контекстный вектор
        context_vector = attention_weights * encoder_outputs
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights

class PointerGeneratorDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        
        # Переключатель: решит, генерировать или копировать
        # Вход: контекст + hidden_state + эмбеддинг
        self.p_gen_linear = nn.Linear(hidden_dim + hidden_dim + emb_dim, 1)
        
        self.lstms = nn.ModuleList([
            nn.LSTM(emb_dim + hidden_dim if i == 0 else hidden_dim, 
                    hidden_dim, batch_first=True) 
            for i in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_states, encoder_outputs, src_indices, src_mask):
        # src_indices: [batch, src_len] — исходные индексы входной строки
        
        embedded = self.embedding(x) 
        last_hidden = hidden_states[-1][0].squeeze(0) 
        context_vector, attn_dist = self.attention(last_hidden, encoder_outputs, src_mask)
        # attn_dist: [batch, src_len, 1] — куда модель "смотрит" прямо сейчас
        
        # 1. Считаем P_gen (вероятность генерации)
        p_gen_input = torch.cat((context_vector, last_hidden, embedded.squeeze(1)), dim=1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)) # число от 0 до 1

        # 2. Обычный проход через LSTM
        lstm_input = torch.cat((context_vector.unsqueeze(1), embedded), dim=2)
        new_hidden_states = []
        for i, lstm in enumerate(self.lstms):
            residual = lstm_input
            lstm_input, new_h = lstm(lstm_input, hidden_states[i])
            new_hidden_states.append(new_h)
            # Residual connection делаем только начиная со ВТОРОГО слоя, 
            # где вход и выход имеют одинаковый размер (hidden_dim)
            if i > 0: 
                lstm_input = self.dropout(lstm_input + residual)
            else:
                lstm_input = self.dropout(lstm_input)

        # 3. Распределение вероятностей словаря (Генерация)
        combined_output = torch.cat((lstm_input.squeeze(1), context_vector), dim=1)
        vocab_dist = F.softmax(self.fc_out(combined_output), dim=1) 
        
        # Взвешиваем вероятности
        vocab_dist_weighted = vocab_dist * p_gen
        
        # 4. Распределение копирования (Pointer)
        # Мы берем веса внимания и "размазываем" их по индексам входной строки
        attn_dist = attn_dist.squeeze(2) # [batch, src_len]
        copy_dist_weighted = attn_dist * (1 - p_gen)
        
        # 5. Финальное объединение
        # Создаем итоговый вектор вероятностей размером с vocab_size
        final_dist = vocab_dist_weighted.clone()
        
        # Добавляем вероятности копирования к соответствующим индексам
        # Например, если мы смотрим на символ с индексом 12 (буква 'e'), 
        # мы добавим вес внимания к 12-й позиции в итоговом векторе.
        final_dist.scatter_add_(1, src_indices, copy_dist_weighted)
        
        return torch.log(final_dist + 1e-12), new_hidden_states
