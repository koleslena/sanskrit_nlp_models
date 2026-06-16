import pandas as pd
import torch
from torch.hub import load_state_dict_from_url

import re

from config import Config
from pos_taggers import bilstm_pos_tagger
from segmenter.segmenter import SanskritPointerSegmenter

DEFAULT_SEGMENTER_MODEL_NAME = 'segmenter_model'
DEFAULT_TAGGER_MODEL_NAME = 'pos_tagger_model'

with_all_bi = ['segmenter_model_1781067108.6164422', 
               'segmenter_model_1781026960.6783192', 
               'segmenter_model_1781091384.9032512', 
               'segmenter_model_1781177574.6417813',
               'segmenter_model_1781204808.2988951',
               'segmenter_model_1781416737.5019588',
               'segmenter_model_1781342500.4066458',
               'segmenter_model_1781270562.565004',
               'segmenter_model_1781237149.2671828',
               'segmenter_model_1781475315.064929']

def load_tagger_model(path, device):
    # 1. Загружаем весь объект
    checkpoint = torch.load(path, map_location=device)
    
    # 2. Извлекаем словари
    char2id = checkpoint['char2id']
    unique_tags = checkpoint['unique_tags']
    config = checkpoint['config']
    
    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = bilstm_pos_tagger.get_model(len(char2id), 
                                        len(unique_tags), 
                                        embedding_size=config['emb_dim'], 
                                        hidden_dim=config['hidden_dim'],
                                        n_layers=config['n_layers'],
                                        research_version=config.get('layer_norm', False),
                                        use_boundary_features=config.get('use_boundary_features', False),
                                        use_char_cnn=config.get("use_char_cnn", False))
    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.char2id = char2id
    model.unique_tags = unique_tags
    model.config = config
    
    return model

def load_segmenter_model(path, device, all_bi=False):
    # 1. Загружаем весь объект
    checkpoint = torch.load(path, map_location=device)
    
    # 2. Извлекаем словари
    char2id = checkpoint['char2id']
    config = checkpoint['config']
    n_layers_dec = config.get('n_layers_dec', -1)
    enc_all_bi = config.get('enc_all_bi', all_bi)
    
    state_dict = checkpoint['model_state_dict']

    if n_layers_dec == -1:
        decoder_layer_indices = []
        for key in state_dict.keys():
            # Регулярка ищет число между 'decoder.lstms.' и следующим за ним элементом
            match = re.match(r"decoder\.lstms\.(\d+)\.", key)
            if match:
                decoder_layer_indices.append(int(match.group(1)))
        # Количество слоев — это максимальный индекс + 1
        n_layers_dec = max(decoder_layer_indices) + 1

    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = SanskritPointerSegmenter(len(char2id), 
                                     config['emb_dim'],
                                     device, 
                                     hidden_dim=config['hidden_dim'], 
                                     n_layers=config['n_layers'],
                                     n_layers_dec=n_layers_dec,
                                     all_bi = enc_all_bi,
                                     with_penalty=config.get('with_penalty', False))
    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(state_dict)
    model.to(device)

    model.char2id = char2id
    model.config = config
    
    return model

def load_tagger_model_from_url(version, device, model_name=DEFAULT_TAGGER_MODEL_NAME):
    # 1. Загружаем весь объект
    if version == 'latest':
        # Читаем CSV прямо из репозитория GitHub
        csv_url = f"https://raw.githubusercontent.com/{Config.repo_name}/main/releases.csv"
        df = pd.read_csv(csv_url)
        df = df[df['url'].str.contains(model_name, na=False)]
        # Берем URL из последней строки
        url = df.iloc[-1]['url']
        print(f"--- Fetching latest model from: {df.iloc[-1]['tag']} ---")
    else:
        url = f"https://github.com/{Config.repo_name}/releases/download/{version}/{model_name}.pth"
    
    # Скачивание и загрузка в память
    checkpoint = load_state_dict_from_url(url, map_location=device, progress=True)
    
    # 2. Извлекаем словари
    char2id = checkpoint['char2id']
    unique_tags = checkpoint['unique_tags']
    config = checkpoint['config']
    
    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = bilstm_pos_tagger.get_model(len(char2id), 
                                        len(unique_tags), 
                                        embedding_size=config['emb_dim'], 
                                        hidden_dim=config['hidden_dim'], 
                                        n_layers=config['n_layers'],
                                        research_version=config.get('layer_norm', False),
                                        use_boundary_features=config.get('use_boundary_features', False),
                                        use_char_cnn=config.get("use_char_cnn", False))

    # 4. Загружаем веса в созданную модель
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.char2id = char2id
    model.unique_tags = unique_tags
    model.config = config
    
    return model

def load_segmenter_model_from_url(version, device, model_name=DEFAULT_SEGMENTER_MODEL_NAME):
    # 1. Загружаем весь объект
    if version == 'latest':
        # Читаем CSV прямо из репозитория GitHub
        csv_url = f"https://raw.githubusercontent.com/{Config.repo_name}/main/releases.csv"
        df = pd.read_csv(csv_url)
        df = df[df['url'].str.contains(model_name, na=False)]
        # Берем URL из последней строки
        url = df.iloc[-1]['url']
        print(f"--- Fetching latest model from: {df.iloc[-1]['tag']} ---")
    else:
        url = f"https://github.com/{Config.repo_name}/releases/download/{version}/{model_name}.pth"
    
    # Скачивание и загрузка в память
    checkpoint = load_state_dict_from_url(url, map_location=device, progress=True)
    
    all_bi = sum([1 for v in with_all_bi if v in url]) == 1

    # 2. Извлекаем словари
    char2id = checkpoint['char2id']
    config = checkpoint['config']
    n_layers_dec = config.get('n_layers_dec', -1)
    enc_all_bi = config.get('enc_all_bi', all_bi)

    state_dict = checkpoint['model_state_dict']

    # TODO почему-то в весах длина словаря + 1, пока так, потом надо разобраться
    saved_vocab_size = state_dict['decoder.fc_out.weight'].shape[0]

    if n_layers_dec == -1:
        decoder_layer_indices = []
        for key in state_dict.keys():
            # Регулярка ищет число между 'decoder.lstms.' и следующим за ним элементом
            match = re.match(r"decoder\.lstms\.(\d+)\.", key)
            if match:
                decoder_layer_indices.append(int(match.group(1)))
        # Количество слоев — это максимальный индекс + 1
        n_layers_dec = max(decoder_layer_indices) + 1
    
    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = SanskritPointerSegmenter(saved_vocab_size, 
                                     config['emb_dim'],
                                     device, 
                                     hidden_dim=config['hidden_dim'], 
                                     n_layers=config['n_layers'],
                                     n_layers_dec=n_layers_dec,
                                     all_bi = enc_all_bi,
                                     with_penalty=config.get('with_penalty', False))

    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(state_dict)
    model.to(device)

    model.char2id = char2id
    model.config = config
    
    return model