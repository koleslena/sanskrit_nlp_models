import torch.nn as nn

import numpy as np


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size, hidden_dim, layers_n, dropout):
        super().__init__()

        self.labels_num = labels_num
        self.hidden_dim = hidden_dim
        
        # 1. Уровень букв (Морфология)
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.char_lstm = nn.LSTM(embedding_size, 
                                 hidden_dim, 
                                 num_layers=layers_n, 
                                 bidirectional=True, 
                                 dropout=dropout,
                                 batch_first=True)
        
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        
        # 2. Уровень предложения (Синтаксис и омонимы)
        # Входной размер равен выходу char_lstm после пуллинга (hidden_dim * 2)
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, 
                                     hidden_dim, 
                                     num_layers=layers_n, 
                                     bidirectional=True, 
                                     dropout=dropout,
                                     batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, labels_num)
        
    def forward(self, tokens):
        """tokens: BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        
        # --- ШАГ 1: Обработка букв ---
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)
        char_embs = self.char_embeddings(tokens_flat) 
        
        # Выделяем признаки из букв каждого слова независимо
        char_feats, _ = self.char_lstm(char_embs)
        char_feats = char_feats.permute(0, 2, 1)  
        
        # Пуллинг дает "сгусток" смысла слова (его морфологический профиль)
        word_vectors = self.global_pooling(char_feats).squeeze(-1) 
        word_vectors = self.dropout(word_vectors)
        
        # --- ШАГ 2: Обработка контекста предложения ---
        # Возвращаем структуру предложения: (BatchSize, MaxSentenceLen, HiddenDim*2)
        sent_context = word_vectors.view(batch_size, max_sent_len, -1)
        
        # Теперь BiLSTM проходит ПО СЛОВАМ в предложении. 
        # Здесь учитываются омонимы и синтаксис!
        sent_feats, _ = self.sentence_lstm(sent_context)
        
        # --- ШАГ 3: Классификация ---
        logits = self.fc(self.dropout(sent_feats)) # (BatchSize, MaxSentenceLen, LabelsNum)
        
        # Возвращаем в формате (BatchSize, LabelsNum, MaxSentenceLen) для CrossEntropy
        return logits.permute(0, 2, 1)

def get_model(vocab_size, labels_num, embedding_size=64, hidden_dim=64, layers_n=4, dropout=0.3):
    bilstm_pos_tagger_model = BiLSTMTagger(vocab_size,
                                        labels_num, 
                                        embedding_size=embedding_size,
                                        hidden_dim=hidden_dim,
                                        layers_n=layers_n, 
                                        dropout=dropout)
    print(f'BLSTM: vocab_size: {vocab_size}, labels_num: {labels_num}')
    print('Количество параметров', sum(np.prod(t.shape) for t in bilstm_pos_tagger_model.parameters()))
    return bilstm_pos_tagger_model

def get_model_name(full_pos=False):
    return "bilstm_pos_tagger_model" if full_pos else "bilstm_pos_tagger_model"