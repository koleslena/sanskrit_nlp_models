from os.path import join, exists
from os import mkdir
import array as arr
import time
import csv

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from sanskrit_tagger.pos_tagger import get_device, copy_data_to_device

from common.pos_datasets import INDEX_PAD

_log_file_name = 'training_log.csv'

class Trainer:
    def __init__(self, datasets, 
                 model, 
                 criterion, 
                 output_path='output',
                 output_model_name='model', 
                 lr=1e-4, epoch_n=10,
                 device=None, 
                 with_metrics=True,
                 metrics_path='metrics',
                 early_stopping_patience=10, 
                 l2_reg_alpha=0,
                 max_batches_per_epoch_train=10000,
                 max_batches_per_epoch_val=1000,
                 optimizer_ctor=None,
                 lr_scheduler_ctor=None):
        
        self.datasets = datasets
        self.model = model
        self.best_val_loss = float('inf')
        self.criterion = criterion
        self.output_path = output_path
        self.output_model_name = output_model_name
        self.lr = lr
        self.epoch_n = epoch_n 
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg_alpha = l2_reg_alpha
        self.max_batches_per_epoch_train = max_batches_per_epoch_train
        self.max_batches_per_epoch_val = max_batches_per_epoch_val
        self.optimizer_ctor = optimizer_ctor
        self.lr_scheduler_ctor = lr_scheduler_ctor
        self.with_metrics = with_metrics
        self.metrics_path = metrics_path

        if self.with_metrics:
            if not exists(self.metrics_path):
                mkdir(self.metrics_path)
            current_path = join(self.metrics_path, f'{self.output_model_name}_metrics_{time.time()}')
            if not exists(current_path):
                mkdir(current_path)
            self.current_path = current_path
            with open(join(self.current_path, _log_file_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_f1'])

        if self.optimizer_ctor is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg_alpha)
        else:
            self.optimizer = self.optimizer_ctor(self.model.parameters(), lr=self.lr)

        # device
        self.device = get_device(device)
        print("using device: ", self.device)
        self.model.to(self.device)

    def _train_epoch(self, msg_format):

        self.model.train()
        total_train_loss = 0
        total_tokens = 0 # Считаем реальные слова, а не батчи

        bar = tqdm(enumerate(self.datasets.train_dataloader), total=len(self.datasets.train_dataloader))

        for batch_i, (batch_x, batch_y) in bar:
            if batch_i > self.max_batches_per_epoch_train:
                break

            batch_x = copy_data_to_device(batch_x, self.device)
            batch_y = copy_data_to_device(batch_y, self.device)

            # Считаем количество реальных слов в батче (не паддингов)
            non_pad_mask = (batch_y != INDEX_PAD)
            num_tokens = non_pad_mask.sum().item()

            pred = self.model(batch_x)
            loss = self.criterion(pred, batch_y)

            self.model.zero_grad()
            loss.backward()

            self.optimizer.step()

            # Накапливаем loss взвешенно (пропорционально количеству слов в батче)
            total_train_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            bar.set_description(msg_format.format(loss.item()))

        return total_train_loss / total_tokens if total_tokens > 0 else 0

    def _val(self):
        self.model.eval()
        total_val_loss = 0
        total_tokens = 0

        # Списки для сбора всех предсказаний и ответов по всей валидации
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(self.datasets.val_dataloader):
                if batch_i > self.max_batches_per_epoch_val:
                    break

                batch_x = copy_data_to_device(batch_x, self.device)
                batch_y = copy_data_to_device(batch_y, self.device)

                # Маска для исключения паддинга (индекс 0)
                mask = (batch_y != INDEX_PAD)
                num_tokens = mask.sum().item()

                # pred shape: (Batch, Labels, Seq)
                pred_logits = self.model(batch_x)
                loss = self.criterion(pred_logits, batch_y)

                # Собираем статистику по лоссу
                total_val_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # Получаем индексы самых вероятных классов: (Batch, Seq)
                pred_classes = torch.argmax(pred_logits, dim=1)
                
                # Собираем данные (переводим в CPU и numpy)
                all_preds.extend(pred_classes[mask].cpu().numpy())
                all_labels.extend(batch_y[mask].cpu().numpy())

        # Средний лосс по всем словам
        mean_val_loss = total_val_loss / total_tokens if total_tokens > 0 else 0
        # Считаем Macro-F1 по всем собранным данным
        mean_val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return mean_val_loss, mean_val_f1
        

    def train(self, epoch_n=None, save_after_train=True):
        if epoch_n is None:
            epoch_n = self.epoch_n

        if self.lr_scheduler_ctor is not None:
            lr_scheduler = self.lr_scheduler_ctor(self.optimizer)
        else:
            lr_scheduler = None
    
        best_epoch_i = 0

        for epoch in range(epoch_n):
            # train
            mean_train_loss = self._train_epoch(f'train {epoch}/{epoch_n} -- loss: {{:3.4f}}')
            print('Среднее значение функции потерь на обучении', mean_train_loss)

            # valuation
            mean_val_loss, mean_val_f1 = self._val()
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            if self.with_metrics:
                self._save_metrics(epoch, mean_train_loss, mean_val_loss, mean_val_f1)

            if mean_val_loss < self.best_val_loss:
                best_epoch_i = epoch
                self.best_val_loss = mean_val_loss
                print('Новая лучшая модель!')
                self._save_model_to_file(self.output_path, f'{self.output_model_name}_tmp.pth')
            elif epoch - best_epoch_i > self.early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    self.early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

        if save_after_train:
            self.save_model()

    def save_model(self):
        self._save_model_to_file(self.output_path, self.output_model_name)

    def _save_model_to_file(self, path, file):
        if not exists(path):
            mkdir(path)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "char2id": self.datasets.char2id,
            "unique_tags": self.datasets.unique_tags,
            "config": {
                "emb_dim": self.model.embedding_size,
                "hidden_dim": self.model.hidden_dim,
                "n_layers": self.model.n_layers,
            }
        }
        
        torch.save(checkpoint, join(path, f'{file}.pth'))

        print(f'Лучшая модель сохранена в {path}!')    

    def _save_metrics(self, epoch, train_loss, val_loss, val_f1):
        with open(join(self.current_path, _log_file_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_f1])