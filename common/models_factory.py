import torch

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