import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout, all_bi=False):
        super(BiLSTMEncoder, self).__init__()

        self.all_bi = all_bi

        # Защита на случай, если hidden_dim случайно передадут нечетным
        assert hidden_dim % 2 == 0, "hidden_dim должен быть четным числом для балансировки BiLSTM"
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # 1-й слой: принимает эмбеддинги, выдает hidden_dim // 2 в каждую сторону (итог: hidden_dim)
        self.first_layer = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        if all_bi:
            # Остальные слои: Двунаправленные с Residual Connections
            # Принимают склеенный hidden_dim от предыдущего слоя, 
            # а выдают снова по hidden_dim // 2 в каждую сторону (итог: hidden_dim)
            self.lstms = nn.ModuleList([
                nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True) 
                for _ in range(n_layers - 1)
            ])
        else:    
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
        
        if self.all_bi:
            max_len = x.shape[1] # Сохраняем исходную длину для распаковки
            # --- СЛОЙ 1 ---
            # Упаковываем тензор: убираем паддинги из вычислений LSTM
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, mask, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.first_layer(packed_x)
            # Распаковываем обратно в обычный тензор, возвращая паддинги на место
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)
            
            # --- ГЛУБОКИЕ СЛОИ С RESIDUAL CONNECTIONS ---
            for lstm in self.lstms:
                residual = x 
                
                # Снова упаковываем перед проходом через очередной BiLSTM
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, mask, batch_first=True, enforce_sorted=False
                )
                packed_out, _ = lstm(packed_x)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)
                
                # Складываем с residual (теперь они оба чистые и совпадают по форме)
                x = self.dropout(x + residual)
        else:         
            x, _ = self.first_layer(x)
            for lstm in self.lstms:
                residual = x
                x, _ = lstm(x)
                x = self.dropout(x + residual)
            
        # x: [batch, seq_len, hidden_dim]
        return x, mask # возвращаем еще и длины для удобства