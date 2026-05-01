import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # 1-й слой: Двунаправленный
        self.first_layer = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        # Остальные слои: Однонаправленные с Residual Connections
        self.lstms = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True) 
            for _ in range(n_layers - 1)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Создаем маску: True там, где НЕ паддинг (индекс 0)
        mask = (x != 0).long().sum(dim=1).cpu() 
        
        x = self.embedding(x)
        
        x, _ = self.first_layer(x)
        
        for lstm in self.lstms:
            residual = x
            x, _ = lstm(x)
            x = self.dropout(x + residual)
            
        # x: [batch, seq_len, hidden_dim]
        return x, mask # возвращаем еще и длины для удобства