import os
from os.path import join, exists
import pickle
from os import mkdir
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

import numpy as np
from sklearn.model_selection import train_test_split

from common.conllu_util import get_files_conllu, read_pos_conllu_file
from common.vocab_util import build_akshara_vocabulary
from common.sanskrit_texts import TEXTS_DIR

INDEX_PAD = -100

class PosSanskritDataset(Dataset):
    def __init__(self, df_data, char2id, label2id):
        # Группируем данные по предложениям (sent_id)
        # Каждое предложение — это список списков (слова из букв)
        grouped = df_data.groupby('sent_id')
        self.sentences = []
        self.labels = []
        
        for name, group in grouped:
            # Превращаем слова в списки ID букв
            sent_chars = [[char2id.get(c, 0) for c in form] for form in group['form_slp1']]
            sent_labels = [label2id.get(l, 0) for l in group['pos']]

            # Проверяем, что в предложении есть хотя бы одно непустое слово
            if any(len(w) > 0 for w in sent_chars):
                self.sentences.append(sent_chars)
                self.labels.append(sent_labels)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


def pos_collate_fn(batch):
    # batch — это список кортежей [(sent, labels), ...]
    sentences, labels = zip(*batch)
    
    # 1. Считаем макс. длины в ЭТОМ батче
    batch_max_sent_len = max(len(s) for s in sentences)
    batch_max_token_len = max(max(len(w) for w in s) for s in sentences)
    
    # 2. Формируем тензоры
    # Наполняем нулями (паддинг)
    inputs = torch.zeros((len(batch), batch_max_sent_len, batch_max_token_len), dtype=torch.long)
    targets = torch.full((len(batch), batch_max_sent_len), INDEX_PAD, dtype=torch.long)
    
    
    for i, (sent, label_seq) in enumerate(zip(sentences, labels)):
        targets[i, :len(label_seq)] = torch.tensor(label_seq)
        for j, word in enumerate(sent):
            inputs[i, j, :len(word)] = torch.tensor(word)
            
    return inputs, targets

class DynamicLengthGroupedSampler(Sampler):
    def __init__(self, dataset, max_tokens, rare_threshold=10, oversample_factor=3):
        self.dataset = dataset
        self.max_tokens = max_tokens
        # Считаем длины (кол-во слов) каждого предложения
        self.lengths = [len(s) for s in dataset.sentences]
        # 2. Считаем глобальные частоты классов именно в этом датасете
        all_labels = [lbl for label_seq in dataset.labels for lbl in label_seq]
        class_counts = Counter(all_labels)
        
        # 3. Формируем расширенный список индексов
        expanded_indices = []
        for idx, label_seq in enumerate(dataset.labels):
            expanded_indices.append(idx)
            
            # Проверяем, есть ли в текущем предложении хоть один редкий тег
            # (но игнорируем <NOTAG>, если у него id=0)
            has_rare_class = any(class_counts[lbl] < rare_threshold for lbl in label_seq if lbl != 0)
            
            if has_rare_class:
                # Добавляем дубликаты этого предложения
                expanded_indices.extend([idx] * (oversample_factor - 1))
        
        # 4. Сортируем ВСЕ индексы (включая дубликаты) по длине предложений.
        # Это сохраняет оптимизацию паддингов! Дубликаты встанут в батчи вместе.
        self.indices = sorted(expanded_indices, key=lambda x: self.lengths[x])

    def __iter__(self):
        batch = []
        current_batch_tokens = 0
        
        for idx in self.indices:
            sent_len = self.lengths[idx]
            
            # Если добавление этого предложения превысит лимит токенов в батче
            if current_batch_tokens + sent_len > self.max_tokens and batch:
                yield batch
                batch = []
                current_batch_tokens = 0
            
            batch.append(idx)
            current_batch_tokens += sent_len
            
        if batch:
            yield batch

    def __len__(self):
        # Примерная оценка кол-ва батчей
        return sum(self.lengths) // self.max_tokens

def get_info(row):
    if row['feats']:
        tags = row['feats']
        return ' '.join([row['upos']] + [tag for tag in tags.values()])
    return row['upos']

class PosDataloaders():
    def __init__(self, 
                 text_files, 
                 max_tokens=800,
                 dataloader_workers_n=0,
                 save_data=False):
        
        files = [file for text in text_files for file in get_files_conllu(f"{TEXTS_DIR}/{text}/")]

        df, sentences = read_pos_conllu_file(files)

        # Фильтруем: UPOS не равен '_' И FORM не равна '_'
        df_clean = df[(df['upos'] != '_') & (df['form'] != '_')].copy()

        # Оставляем только те строки, где форма слова реально содержит символы
        df_clean = df_clean[df_clean['form'].str.strip().str.len() > 0].copy()
        df_clean = df_clean.reset_index()

        char_tokenized = [list(token) for sent in sentences for token in sent.split()]
        self.char2id = build_akshara_vocabulary(char_tokenized, pad_word='<PAD>')

        df_clean['pos'] = df_clean.apply(lambda row: get_info(row).rstrip(), axis=1)

        self.unique_tags = ['<NOTAG>'] + sorted(df_clean['pos'].value_counts().index)
        label2id = {label: i for i, label in enumerate(self.unique_tags)}

        self.vocab_size = len(self.char2id)
        self.labels_num = len(self.unique_tags)

        unique_sents = df_clean['sent_id'].unique()
        train_sents, val_sents = train_test_split(unique_sents, test_size=0.2, random_state=42)

        df_train = df_clean[df_clean['sent_id'].isin(train_sents)]
        df_val = df_clean[df_clean['sent_id'].isin(val_sents)]

        # Считаем частоты для каждого тега, который есть в df_train
        train_counts = df_train['pos'].value_counts()

        # Создаем массив весов размером с количество классов
        weights = np.ones(self.labels_num, dtype=np.float32)

        # Константа сглаживания (чем она больше, тем ближе веса к 1.0)
        # 1.5 или 2.0 — идеальный баланс для 807 классов
        C = 1.5

        for label, idx in label2id.items():
            if label == '<NOTAG>':
                weights[idx] = 1.0
                continue
                
            if label in train_counts:
                count = train_counts[label]
                # Мягкое логарифмическое сглаживание
                weights[idx] = 1.0 / np.sqrt(count)
            else:
                # Для экстремально редких тегов, которых вообще нет в трейне
                weights[idx] = 1.0 / np.sqrt(1)

        # Нормализуем веса, чтобы среднее было около 1.0
        weights = weights / np.mean(weights)
        self.class_weights = torch.FloatTensor(weights)

        # Создаем датасеты
        train_ds = PosSanskritDataset(df_train, self.char2id, label2id)
        val_ds = PosSanskritDataset(df_val, self.char2id, label2id)

        # Создаем список батчей через сэмплер
        train_sampler = DynamicLengthGroupedSampler(
            train_ds, 
            max_tokens=max_tokens, 
            rare_threshold=200, 
            oversample_factor=50
        )
        train_list_of_batches = list(train_sampler)
        random.shuffle(train_list_of_batches)

        # DataLoader
        self.train_dataloader = DataLoader(
            train_ds, 
            batch_sampler=train_list_of_batches, 
            collate_fn=pos_collate_fn,
            num_workers=dataloader_workers_n
        )

        val_sampler = DynamicLengthGroupedSampler(
            val_ds, 
            max_tokens=max_tokens, 
            rare_threshold=0, 
            oversample_factor=1
        )
        val_list_of_batches = list(val_sampler)

        self.val_dataloader = DataLoader(
            val_ds, 
            batch_sampler=val_list_of_batches,
            shuffle=False,
            collate_fn=pos_collate_fn,
            num_workers=dataloader_workers_n
        )

        if save_data:
            self.save_data()

    def save_data(self, output_path='output'):
        if not exists(output_path):
            mkdir(output_path)
        with open(join(output_path, f'unique_tags_{self.vocab_size}_{self.labels_num}.dat'), 'wb') as file:
            pickle.dump(self.unique_tags, file)

        with open(join(output_path, f'char2id_{self.vocab_size}_{self.labels_num}.dat'), 'wb') as file:
            pickle.dump(self.char2id, file)