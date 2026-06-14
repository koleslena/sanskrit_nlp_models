import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Слой для вычисления весов внимания
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: (Batch*Sent, MaxTokenLen, HiddenDim)
        weights = self.attention(x).squeeze(-1) # (Batch*Sent, MaxTokenLen)
        
        if mask is not None:
            # Зануляем влияние паддингов (букв-нулей)
            weights = weights.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(weights, dim=-1)
        # Взвешенная сумма векторов букв
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return weighted

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size, hidden_dim, n_layers, dropout, research_version=False, use_boundary_features=False):
        super().__init__()

        self.labels_num = labels_num
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.use_boundary_features = use_boundary_features
        
        # 1. Уровень букв (Морфология)
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.char_lstm = nn.LSTM(embedding_size, 
                                 hidden_dim, 
                                 num_layers=n_layers, 
                                 bidirectional=True, 
                                 dropout=dropout,
                                 batch_first=True)
        
        # Заменяем MaxPool на Attention
        self.char_attention = AttentionPooling(hidden_dim * 2)

        # включает суффиксы и префиксы
        if self.use_boundary_features:
            self.feat_emb_dim = 32 
            self.pref1_emb = nn.Embedding(vocab_size, self.feat_emb_dim, padding_idx=0)
            self.pref2_emb = nn.Embedding(vocab_size, self.feat_emb_dim, padding_idx=0)
            self.suff1_emb = nn.Embedding(vocab_size, self.feat_emb_dim, padding_idx=0)
            self.suff2_emb = nn.Embedding(vocab_size, self.feat_emb_dim, padding_idx=0)
            
            extra_features_dim = 4 * self.feat_emb_dim
            self.feature_mixing = nn.Linear(hidden_dim * 2 + extra_features_dim, hidden_dim * 2)
        
        # 2. Уровень предложения (Синтаксис и омонимы)
        # Входной размер равен выходу char_lstm после пуллинга (hidden_dim * 2)
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, 
                                     hidden_dim, 
                                     num_layers=n_layers, 
                                     bidirectional=True, 
                                     dropout=dropout,
                                     batch_first=True)
        
        # Фикс баланса масштабов: нормализация после сложения ресидуала
        self.layer_norm = nn.LayerNorm(hidden_dim * 2) if research_version else None

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
        char_lengths = char_mask.sum(dim=-1).cpu().clamp(min=1)

        char_embs = self.char_embeddings(tokens_flat)
        
        # Выделяем признаки из букв каждого слова независимо
        # Упаковка букв (чтобы не считать LSTM на паддингах внутри слова)
        packed_chars = nn.utils.rnn.pack_padded_sequence(
            char_embs, char_lengths, batch_first=True, enforce_sorted=False
        )
        packed_feats, _ = self.char_lstm(packed_chars)
        char_feats, _ = nn.utils.rnn.pad_packed_sequence(
            packed_feats, batch_first=True, total_length=max_token_len
        )
        
        # Используем Attention вместо GlobalPooling
        word_vectors = self.char_attention(char_feats, mask=char_mask)
        word_vectors = self.dropout(word_vectors)

        # --- Применение фич по условию суффиксы и префиксы ---
        if self.use_boundary_features:
            N_flat = tokens_flat.size(0)
            device = tokens.device
            
            p1 = tokens_flat[:, 0]
            p2 = tokens_flat[:, 1] if max_token_len > 1 else torch.zeros_like(p1)
            
            idx_last = (char_lengths - 1).to(device)
            idx_penultimate = (char_lengths - 2).clamp(min=0).to(device)
            
            s1 = tokens_flat[torch.arange(N_flat, device=device), idx_last]
            s2 = tokens_flat[torch.arange(N_flat, device=device), idx_penultimate]
            condition = (char_lengths > 1).to(device)
            s2 = torch.where(condition, s2, torch.zeros_like(s2))
            
            p1_e = self.pref1_emb(p1)
            p2_e = self.pref2_emb(p2)
            s1_e = self.suff1_emb(s1)
            s2_e = self.suff2_emb(s2)
            
            anchored_word_vectors = torch.cat([word_vectors, p1_e, p2_e, s1_e, s2_e], dim=-1)
            word_vectors = self.feature_mixing(anchored_word_vectors)

        # --- ШАГ 2: Предложение с Residual ---
        sent_context = word_vectors.view(batch_size, max_sent_len, -1)

        packed_sent = nn.utils.rnn.pack_padded_sequence(
            sent_context, sent_lengths, batch_first=True, enforce_sorted=False
        )
                
        # Теперь BiLSTM проходит ПО СЛОВАМ в предложении. 
        # Здесь учитываются омонимы и синтаксис!
        packed_output, _ = self.sentence_lstm(packed_sent)
        sent_feats, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_sent_len
        )

        # RESIDUAL CONNECTION: Добавляем входные векторы слов к выходу LSTM
        # Это помогает модели не "забывать" морфологию букв при анализе синтаксиса
        # ФИКС: Обнуляем мусорные векторы паддингов перед ресидуалом
        word_mask_expanded = word_mask.unsqueeze(-1).float()
        sent_context = sent_context * word_mask_expanded
        
        if self.layer_norm: 
            # Складываем и пропускаем через LayerNorm
            sent_feats = self.layer_norm(sent_feats + sent_context) * word_mask_expanded
        else:
            sent_feats = sent_feats + sent_context             

        # --- ШАГ 3: Классификация ---
        logits = self.fc(self.dropout(sent_feats))
        # Возвращаем в формате (BatchSize, LabelsNum, MaxSentenceLen)
        return logits.permute(0, 2, 1)

def get_model(vocab_size, labels_num, embedding_size=128, hidden_dim=256, n_layers=3, dropout=0.3, research_version=True, use_boundary_features=True):
    bilstm_pos_tagger_model = BiLSTMTagger(vocab_size,
                                        labels_num, 
                                        embedding_size=embedding_size,
                                        hidden_dim=hidden_dim,
                                        n_layers=n_layers, 
                                        dropout=dropout, 
                                        research_version=research_version,
                                        use_boundary_features=use_boundary_features)
    print(f'BLSTM: vocab_size: {vocab_size}, labels_num: {labels_num}')
    print('Количество параметров', sum(np.prod(t.shape) for t in bilstm_pos_tagger_model.parameters()))
    return bilstm_pos_tagger_model

def get_model_name(full_pos=False):
    return "bilstm_pos_tagger_model" if full_pos else "bilstm_pos_tagger_model"