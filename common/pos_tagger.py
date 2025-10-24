import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from common.device_util import get_device, copy_data_to_device

class POSTagger:
    def __init__(self, model, char2id, id2label, max_sent_len, max_token_len, device=None, batch_size=64):
        self.model = model
        self.char2id = char2id
        self.id2label = id2label
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len
        self.device = device
        self.batch_size = batch_size

    def __call__(self, sentences):
        tokenized_corpus = [sent.split() for sent in sentences]

        inputs = torch.zeros((len(sentences), self.max_sent_len, self.max_token_len + 2), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):
                for char_i, char in enumerate(token):
                    inputs[sent_i, token_i, char_i + 1] = self.char2id.get(char, 0)

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_probs = self._predict_with_model(dataset)  # SentenceN x TagsN x MaxSentLen
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append([self.id2label[cls] for cls in predicted_classes[sent_i, :len(sent)]])
        return result
    
    def _predict_with_model(self, dataset, return_labels=False):
        device = get_device(self.device)
        results_by_batch = []

        self.model.to(device)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        labels = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader, total=len(dataset)/self.batch_size):
                batch_x = copy_data_to_device(batch_x, device)

                if return_labels:
                    labels.append(batch_y.numpy())

                batch_pred = self.model(batch_x)
                results_by_batch.append(batch_pred.detach().cpu().numpy())

        if return_labels:
            return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
        else:
            return np.concatenate(results_by_batch, 0)