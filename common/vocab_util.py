import collections
import numpy as np
import torch

def build_akshara_vocabulary(tokenized_texts, after_sort=True, min_count=3, pad_word=None):
    akshara_counts = collections.defaultdict(int)

    for txt in tokenized_texts:
        unique_text_chr = set(txt)
        for chr in unique_text_chr:
            akshara_counts[chr] += 1

   # убрать слишком редкие 
    akshara_counts = {chr: cnt for chr, cnt in akshara_counts.items() if cnt >= min_count}

    akshara_counts[" "] = -1
    akshara_counts["-"] = -1

    # отсортировать по убыванию частоты
    if after_sort:
        sorted_akshara_counts = sorted(akshara_counts.items(),
                                    reverse=True,
                                    key=lambda pair: pair[1])

    # добавим несуществующий символ с индексом 0 для удобства пакетной обработки
    if pad_word:
        sorted_akshara_counts = [(pad_word, 0)] + sorted_akshara_counts
        SOS_token = len(sorted_akshara_counts)
        EOS_token = len(sorted_akshara_counts) + 1
        sorted_akshara_counts = sorted_akshara_counts + [('<SOS>', SOS_token), ('<EOS>', EOS_token)]

    # нумеруем символы
    chr2id = {chr: i for i, (chr, _) in enumerate(sorted_akshara_counts)}

    return chr2id

def build_vocabulary(tokenized_texts, min_count=5, pad_word=None):
    akshara_counts = collections.defaultdict(int)
    doc_n = 0

    # посчитать количество документов, в которых употребляется каждое слово
    # а также общее количество документов
    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            akshara_counts[token] += 1

    # убрать слишком редкие 
    akshara_counts = {chr: cnt for chr, cnt in akshara_counts.items() if cnt >= min_count}

    # отсортировать слова по убыванию частоты
    sorted_akshara_counts = sorted(akshara_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    # добавим несуществующий символ с индексом 0 для удобства пакетной обработки
    if pad_word is not None:
        sorted_akshara_counts = [(pad_word, 0)] + sorted_akshara_counts

    # нумеруем символы
    chr2id = {word: i for i, (word, _) in enumerate(sorted_akshara_counts)}

    return chr2id


def pos_corpus_to_tensor(df_data, char2id, label2id, sent_count, max_sent_len, max_token_len):
    inputs = torch.zeros((sent_count, max_sent_len, max_token_len + 5), dtype=torch.long)
    targets = torch.zeros((sent_count, max_sent_len), dtype=torch.long)

    for row in df_data.values:
        targets[row[0], row[1]] = label2id.get(row[3], 0)
        for char_i, char in enumerate(row[2]):
            inputs[row[0], row[1], char_i + 1] = char2id.get(char, 0)

    return inputs, targets