import torch.nn as nn

import numpy as np


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size, hidden_dim, n_layers, dropout):
        super().__init__()

        self.labels_num = labels_num
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        
        # 1. Уровень букв (Морфология)
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.char_lstm = nn.LSTM(embedding_size, 
                                 hidden_dim, 
                                 num_layers=n_layers, 
                                 bidirectional=True, 
                                 dropout=dropout,
                                 batch_first=True)
        
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        
        # 2. Уровень предложения (Синтаксис и омонимы)
        # Входной размер равен выходу char_lstm после пуллинга (hidden_dim * 2)
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, 
                                     hidden_dim, 
                                     num_layers=n_layers, 
                                     bidirectional=True, 
                                     dropout=dropout,
                                     batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, labels_num)
        
    def forward(self, tokens):
        """tokens: BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        
        # 1. МАСКА: находим, где реально есть слова (не все нули в слове)
        # (batch_size, max_sent_len)
        word_mask = (tokens.sum(dim=-1) != 0) 
        # Длины предложений для упаковки
        sent_lengths = word_mask.sum(dim=-1).cpu().clamp(min=1)

        # --- ШАГ 1: Обработка букв ---
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)
        # Считаем длины слов для упаковки на уровне букв
        char_mask = (tokens_flat != 0)
        char_lengths = char_mask.sum(dim=-1).cpu().clamp(min=1) # минимум 1 символ

        char_embs = self.char_embeddings(tokens_flat)
        
        # Выделяем признаки из букв каждого слова независимо
        # Упаковка букв (чтобы не считать LSTM на паддингах внутри слова)
        packed_chars = nn.utils.rnn.pack_padded_sequence(
            char_embs, char_lengths, batch_first=True, enforce_sorted=False
        )
        packed_feats, _ = self.char_lstm(packed_chars)
        char_feats, _ = nn.utils.rnn.pad_packed_sequence(packed_feats, batch_first=True)
        
        char_feats = char_feats.permute(0, 2, 1)
        # Пуллинг дает "сгусток" смысла слова (его морфологический профиль)
        word_vectors = self.global_pooling(char_feats).squeeze(-1)
        word_vectors = self.dropout(word_vectors)

        # --- ШАГ 2: Обработка контекста предложения ---
        # Возвращаем структуру предложения: (BatchSize, MaxSentenceLen, HiddenDim*2)
        sent_context = word_vectors.view(batch_size, max_sent_len, -1)

        # Упаковка слов (чтобы модель не видела паддинги в конце предложения)
        packed_sent = nn.utils.rnn.pack_padded_sequence(
            sent_context, sent_lengths, batch_first=True, enforce_sorted=False
        )
                
        # Теперь BiLSTM проходит ПО СЛОВАМ в предложении. 
        # Здесь учитываются омонимы и синтаксис!
        packed_output, _ = self.sentence_lstm(packed_sent)
        sent_feats, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_sent_len
        )

        # --- ШАГ 3: Классификация ---
        logits = self.fc(self.dropout(sent_feats))
        # Возвращаем в формате (BatchSize, LabelsNum, MaxSentenceLen)
        return logits.permute(0, 2, 1)

def get_model(vocab_size, labels_num, embedding_size=128, hidden_dim=256, n_layers=3, dropout=0.3):
    bilstm_pos_tagger_model = BiLSTMTagger(vocab_size,
                                        labels_num, 
                                        embedding_size=embedding_size,
                                        hidden_dim=hidden_dim,
                                        n_layers=n_layers, 
                                        dropout=dropout)
    print(f'BLSTM: vocab_size: {vocab_size}, labels_num: {labels_num}')
    print('Количество параметров', sum(np.prod(t.shape) for t in bilstm_pos_tagger_model.parameters()))
    return bilstm_pos_tagger_model

def get_model_name(full_pos=False):
    return "bilstm_pos_tagger_model" if full_pos else "bilstm_pos_tagger_model"