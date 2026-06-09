import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super(BiLSTMEncoder, self).__init__()
        # Защита на случай, если hidden_dim случайно передадут нечетным
        assert hidden_dim % 2 == 0, "hidden_dim должен быть четным числом для балансировки BiLSTM"
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # 1-й слой: принимает эмбеддинги, выдает hidden_dim // 2 в каждую сторону (итог: hidden_dim)
        self.first_layer = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        # Остальные слои: Двунаправленные с Residual Connections
        # Принимают склеенный hidden_dim от предыдущего слоя, 
        # а выдают снова по hidden_dim // 2 в каждую сторону (итог: hidden_dim)
        self.lstms = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True) 
            for _ in range(n_layers - 1)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Создаем маску: True там, где НЕ паддинг (индекс 0)
        mask = (x != 0).long().sum(dim=1).cpu() 
        
        x = self.embedding(x)
        
        # Проход через первый слой -> получаем [batch, seq_len, hidden_dim]
        x, _ = self.first_layer(x)
        
        # Проход через глубокие двунаправленные слои
        for lstm in self.lstms:
            residual = x # Размерность [batch, seq_len, hidden_dim]
            x, _ = lstm(x) # На входе hidden_dim, на выходе снова hidden_dim (из-за bidirectional)
            x = self.dropout(x + residual)
            
        # x: [batch, seq_len, hidden_dim]
        return x, mask # возвращаем еще и длины для удобства