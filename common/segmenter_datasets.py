import json

import torch
from torch.utils.data import DataLoader

from functools import partial
import numpy as np

from sklearn.model_selection import train_test_split

from common.vocab_util import build_akshara_vocabulary
from common.conllu_util import get_files_conllu, read_split_conllu_file
from common.sanskrit_texts import TEXTS_DIR

class AdaptiveTokenSampler(torch.utils.data.Sampler):
    def __init__(self, df, max_tokens_per_batch=2000, shuffle=True):
        self.max_tokens = max_tokens_per_batch
        self.shuffle = shuffle
        
        # Сортируем индексы по длине предложения для эффективной группировки
        lengths = df['src'].str.len().values
        self.sorted_indices = np.argsort(lengths)
        
        self.batches = self._create_batches(lengths)

    def _create_batches(self, lengths):
        batches = []
        current_batch = []
        current_max_len = 0
        
        for idx in self.sorted_indices:
            l = lengths[idx]
            # Ориентируемся на площадь тензора (max_len * num_sentences)
            temp_max_len = max(current_max_len, l)
            if temp_max_len * (len(current_batch) + 1) > self.max_tokens:
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = l
            else:
                current_batch.append(idx)
                current_max_len = temp_max_len
        
        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class SanskritListDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.src_texts = df['src'].values
        self.trg_texts = df['trg'].values

    def __getitem__(self, idx):
        return self.src_texts[idx], self.trg_texts[idx]

    def __len__(self):
        return len(self.src_texts)

def smart_collate(batch, char2id):
    PAD_token = char2id.get('<PAD>', 0)
    SOS_token = char2id.get('<SOS>', len(char2id)-2)
    EOS_token = char2id.get('<EOS>', len(char2id)-1)
    
    src_list, trg_list = zip(*batch)
    
    # Определяем макс. длину в ЭТОМ батче
    max_src = max(len(s) for s in src_list) + 1 # +1 для EOS
    max_trg = max(len(t) for t in trg_list) + 2 # +2 для SOS и EOS
    
    inputs = torch.full((len(batch), max_src), PAD_token, dtype=torch.long)
    targets = torch.full((len(batch), max_trg), PAD_token, dtype=torch.long)
    
    for i, (s_text, t_text) in enumerate(zip(src_list, trg_list)):
        # Заполнение SRC
        for char_i, char in enumerate(s_text):
            inputs[i, char_i] = char2id.get(char, PAD_token)
        inputs[i, len(s_text)] = EOS_token
        
        # Заполнение TRG
        targets[i, 0] = SOS_token
        for char_i, char in enumerate(t_text):
            targets[i, char_i + 1] = char2id.get(char, PAD_token)
        targets[i, len(t_text) + 1] = EOS_token
            
    return inputs, targets

class SegmenterDataloaders:
    def __init__(self, 
                 file_texts, 
                 dataloader_workers_n=0,
                 max_tokens_per_batch=8000,
                 save_data=False):
        
        self.file_texts = file_texts
        
        files = [file for text in self.file_texts for file in get_files_conllu(f"{TEXTS_DIR}/{text}/")]

        df = read_split_conllu_file(files, transliterate=True)

        # Динамический порог фильтрации
        max_sentence_len = (max_tokens_per_batch // 2) - 4
        
        initial_len = len(df)
        df = df[df['src'].str.len() <= max_sentence_len].copy()
        
        print(f"Фильтрация: удалено {initial_len - len(df)} предложений "
              f"длиннее {max_sentence_len} символов.")
        
        all_tokens = [token for list_tokens in [df['trg'].str.split(), df['src'].str.split()] for tokens in list_tokens for token in tokens]

        char_tokenized = [list(token) for token in all_tokens]

        char2id = build_akshara_vocabulary(char_tokenized, pad_word='<PAD>')

        self.vocab_size = len(char2id)
        self.char2id = char2id

        print(f"Словарь: {self.vocab_size}")

        # 1. Создаем датасеты из "сырых" строк (но уже очищенных/обработанных)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        train_dataset = SanskritListDataset(train_df)
        val_dataset = SanskritListDataset(val_df)

        # 2. Создаем самплеры
        # max_tokens_per_batch — это размер "окна" памяти. 
        train_sampler = AdaptiveTokenSampler(train_df, max_tokens_per_batch=max_tokens_per_batch, shuffle=True)
        val_sampler = AdaptiveTokenSampler(val_df, max_tokens_per_batch=max_tokens_per_batch, shuffle=False)

        custom_collate = partial(smart_collate, char2id=char2id)

        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=custom_collate,
            num_workers=dataloader_workers_n
        )

        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_sampler=val_sampler, 
            collate_fn=custom_collate,
            num_workers=dataloader_workers_n
        )

        if save_data:
            with open(f'output/char2id_{len(char2id)}.json', 'w', encoding='utf-8') as f:
                json.dump(char2id, f, ensure_ascii=False, indent=4)
    
    def get_vocab_size(self):
        return self.vocab_size
        
    def get_char2id(self):
        return self.char2id


