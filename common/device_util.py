import torch

def get_device(device=None):
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.4, 0)
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(0.4)
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)
    
def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))