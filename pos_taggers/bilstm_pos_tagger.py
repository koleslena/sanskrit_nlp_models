from torch import nn

import numpy as np


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size, hidden_dim, layers_n, dropout):
        super().__init__()

        self.labels_num = labels_num
        
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, 
                            hidden_dim, 
                            num_layers=layers_n, 
                            bidirectional=True, 
                            dropout=dropout,
                            batch_first=True) # batch_first=True for (batch_size, seq_len, input_size)
        
        self.fc = nn.Linear(hidden_dim * 2, labels_num)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens):
        """tokens - BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)
        
        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        
        features, (hidden, cell) = self.lstm(char_embeddings)
        features = features.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        
        global_features = self.global_pooling(features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize
        out = self.dropout(global_features)

        logits_flat = self.fc(out)
        logits = logits_flat.view(batch_size, max_sent_len, self.labels_num)  # BatchSize x MaxSentenceLen x LabelsNum
        logits = logits.permute(0, 2, 1)  # BatchSize x LabelsNum x MaxSentenceLen
        return logits

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