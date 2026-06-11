import pandas as pd
import torch
from torch.hub import load_state_dict_from_url

from config import Config
from pos_taggers import bilstm_pos_tagger
from segmenter.segmenter import SanskritPointerSegmenter

DEFAULT_SEGMENTER_MODEL_NAME = 'segmenter_model'
DEFAULT_TAGGER_MODEL_NAME = 'pos_tagger_model'

with_all_bi = ['segmenter_model_1781067108.6164422', 
               'segmenter_model_1781026960.6783192', 
               'segmenter_model_1781091384.9032512', 
               'segmenter_model_1781177574.6417813']

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
                                        research_version=config.get('layer_norm', False))
    
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
    
    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = SanskritPointerSegmenter(len(char2id), 
                                     config['emb_dim'],
                                     device, 
                                     hidden_dim=config['hidden_dim'], 
                                     n_layers=config['n_layers'],
                                     all_bi = all_bi,
                                     with_penalty=config.get('with_penalty', False))
    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(checkpoint['model_state_dict'])
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
                                        research_version=config.get('layer_norm', False))

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

    # TODO почему-то в весах длина словаря + 1, пока так, потом надо разобраться
    saved_vocab_size = checkpoint['model_state_dict']['decoder.fc_out.weight'].shape[0]
    
    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = SanskritPointerSegmenter(saved_vocab_size, 
                                     config['emb_dim'],
                                     device, 
                                     hidden_dim=config['hidden_dim'], 
                                     n_layers=config['n_layers'],
                                     all_bi = all_bi,
                                     with_penalty=config.get('with_penalty', False))

    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.char2id = char2id
    model.config = config
    
    return model