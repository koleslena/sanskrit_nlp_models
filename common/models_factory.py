import pandas as pd
import torch
from torch.hub import load_state_dict_from_url

from config import Config
from segmenter.segmenter import SanskritPointerSegmenter

DEFAULT_SEGMENTER_MODEL_NAME = 'segmenter_model'

def load_segmenter_model(path, device):
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
                                     n_layers=config['n_layers'])
    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.char2id = char2id
    model.config = config
    
    return model


def load_segmenter_model_from_url(version, device):
    # 1. Загружаем весь объект
    if version == 'latest':
        # Читаем CSV прямо из репозитория GitHub
        csv_url = f"https://raw.githubusercontent.com/{Config.repo_name}/main/releases.csv"
        df = pd.read_csv(csv_url)
        # Берем URL из последней строки
        url = df.iloc[-1]['url']
        print(f"--- Fetching latest model from: {df.iloc[-1]['tag']} ---")
    else:
        url = f"https://github.com/{Config.repo_name}/releases/download/{version}/segmenter_model.pth"
    
    # Скачивание и загрузка в память
    checkpoint = load_state_dict_from_url(url, map_location=device, progress=True)
    
    # 2. Извлекаем словари
    char2id = checkpoint['char2id']
    config = checkpoint['config']
    
    # 3. Создаем экземпляр модели, используя сохраненный конфиг
    model = SanskritPointerSegmenter(len(char2id), 
                                     config['emb_dim'],
                                     device, 
                                     hidden_dim=config['hidden_dim'], 
                                     n_layers=config['n_layers'])

    
    # 4. Загружаем веса в созданную модель
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.char2id = char2id
    model.config = config
    
    return model