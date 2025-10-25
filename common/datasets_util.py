import os
from os.path import join, exists
import pickle
from os import mkdir

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from enum import Enum

from common.vocab_util import build_vocabulary, pos_corpus_to_tensor
from common.conllu_util import get_files_conllu, read_conllu_file


_TEXTS_DIR = os.environ.get("SANSKRIT_TEXTS_DIR")

class Datasources(str, Enum):
    MAHABHARATA = 'Mahābhārata'
    RAMAYANA = 'Rāmāyaṇa'
    HITOPADESHA = 'Hitopadeśa'
    AMARAKOSHA = 'Amarakośa'

def get_info(row):
    if row['feats']:
        tags = row['feats']
        return ' '.join([row['upos']] + [tag for tag in tags.values()])
    return row['upos']

def get_sent(df, sents_count):
    sents = range(sents_count)
    used_sents = []

    def is_sent_start_samasa(row_id):
        return type(row_id) is tuple and row_id[0] == 1

    def calc(row):
        if row.name == 0:
            used_sents.append(0)
            return sents[0]
        else:
            s = used_sents[len(used_sents) - 1]
            if (row['id'] == 1 and not is_sent_start_samasa(df.loc[row.name - 1, 'id'])) or is_sent_start_samasa(row['id']):
                s += 1
                used_sents.append(s)
            return sents[s]
    
    return df.apply(lambda row: calc(row), axis=1)

def get_sent_for_pos(df, sents_count):
    sents = range(sents_count)
    used_sents = []

    def calc(row):
        if row.name == 0:
            used_sents.append(0)
            return sents[0]
        else:
            s = used_sents[len(used_sents) - 1]
            if row['id'] == 1:
                s += 1
                used_sents.append(s)
            return sents[s]
    
    return df.apply(lambda row: calc(row), axis=1)

def get_pos_datasets(train_texts, val_texts, full_pos=False, **kwargs):
    train_files = [text for texts in train_texts for text in get_files_conllu(f"{_TEXTS_DIR}/{texts}/")]
    val_files = [text for texts in val_texts for text in get_files_conllu(f"{_TEXTS_DIR}/{texts}/")]

    train_df, train_sentences, train_all_tokens = read_conllu_file(train_files)
    val_df, val_sentences, _ = read_conllu_file(val_files)

    train_sent_count = len(train_sentences)
    val_sent_count = len(val_sentences)

    train_sentences_splited = [' '.join([s['form'] for s in sent if str(s['id']).isdigit()]) for sent in train_sentences]

    train_all_tokens_clean = [(token['form'], token['upos'], token['feats']) for token in train_all_tokens if str(token['id']).isdigit()]

    df_train_clean = train_df[train_df['upos'] != '_']
    df_val_clean = val_df[val_df['upos'] != '_']

    df_train_clean['sent'] = get_sent_for_pos(df_train_clean, train_sent_count)
    df_val_clean['sent'] = get_sent_for_pos(df_val_clean, val_sent_count)

    max_sent_len = max(len(sent) for sent in train_sentences_splited)
    max_origin_token_len = max(len(token[0]) for token in train_all_tokens_clean)

    train_char_tokenized = [list(sent) for sent in train_sentences_splited]

    char2id, _ = build_vocabulary(train_char_tokenized, max_doc_freq=1.0, min_count=5, pad_word='<PAD>')

    pos = 'upos'

    if full_pos:
        pos = 'pos'
        df_train_clean['pos'] = df_train_clean.apply(lambda row: get_info(row).rstrip(), axis=1)
        df_val_clean['pos'] = df_val_clean.apply(lambda row: get_info(row).rstrip(), axis=1)

    unique_tags = ['<NOTAG>'] + sorted(df_train_clean[pos].value_counts().index)
    label2id = {label: i for i, label in enumerate(unique_tags)}

    train_inputs, train_labels = pos_corpus_to_tensor(df_train_clean[['sent', 'id', 'form', pos]], char2id, label2id, train_sent_count, max_sent_len, max_origin_token_len)
    train_dataset = TensorDataset(train_inputs, train_labels)

    val_inputs, val_labels = pos_corpus_to_tensor(df_val_clean[['sent', 'id', 'form', pos]], char2id, label2id, val_sent_count, max_sent_len, max_origin_token_len)
    val_dataset = TensorDataset(val_inputs, val_labels)

    return Datasets(train_dataset, val_dataset, char2id, max_sent_len, max_origin_token_len, unique_tags, **kwargs)

class Datasets():
    def __init__(self, train_dataset, val_dataset, 
                 char2id,
                 max_sent_len,
                 max_origin_token_len,
                 unique_tags,
                 data_loader=DataLoader,
                 shuffle_train=True,
                 dataloader_workers_n=0,
                 batch_size=64):
        
        self.max_sent_len = max_sent_len
        self.max_origin_token_len = max_origin_token_len
        self.unique_tags = unique_tags
        self.char2id = char2id
        self.vocab_size = len(char2id)
        self.labels_num = len(unique_tags)

        self.train_dataloader = data_loader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                            num_workers=dataloader_workers_n)
        self.val_dataloader = data_loader(val_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=dataloader_workers_n)
        
    def save_data(self, output_path='output'):
        if not exists(output_path):
            mkdir(output_path)
        with open(join(output_path, f'unique_tags_{self.vocab_size}_{self.labels_num}.dat'), 'wb') as file:
            pickle.dump(self.unique_tags, file)

        with open(join(output_path, f'char2id_{self.vocab_size}_{self.labels_num}.dat'), 'wb') as file:
            pickle.dump(self.char2id, file)
