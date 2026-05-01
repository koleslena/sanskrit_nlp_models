import time
import csv
from os.path import join, exists
from os import mkdir

import torch
import torch.nn as nn
from tqdm import tqdm

from sanskrit_tagger.pos_tagger import get_device, copy_data_to_device

_log_file_name = "segmenter_train_log.csv"

class SegmenterTrainer:
    def __init__(self, 
                 datasets, 
                 model, 
                 criterion=nn.NLLLoss(ignore_index=0),  # Игнорируем <PAD>
                 output_path='segmenter_output',
                 output_model_name='segmenter_model', 
                 lr=0.001, 
                 epoch_n=10,
                 device=None, 
                 save_metrics=True,
                 metrics_path='metrics',
                 early_stopping_patience=10, 
                 max_batches_per_epoch_train=100000,
                 max_batches_per_epoch_val=10000,
                 optimizer_ctor=None):
        
        self.datasets = datasets
        self.model = model
        self.best_model = model
        self.best_val_loss = float('inf')
        self.criterion = criterion
        self.output_path = output_path
        self.output_model_name = output_model_name
        self.lr = lr
        self.epoch_n = epoch_n 
        self.early_stopping_patience = early_stopping_patience
        self.max_batches_per_epoch_train = max_batches_per_epoch_train
        self.max_batches_per_epoch_val = max_batches_per_epoch_val
        self.optimizer_ctor = optimizer_ctor
        self.save_metrics = save_metrics
        self.metrics_path = metrics_path

        if self.save_metrics:
            if not exists(self.metrics_path):
                mkdir(self.metrics_path)
            current_path = join(self.metrics_path, f'{self.output_model_name}_metrics_{time.time()}')
            if not exists(current_path):
                mkdir(current_path)
            self.current_path = current_path
            with open(join(self.current_path, _log_file_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])

        if self.optimizer_ctor is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = self.optimizer_ctor(self.model.parameters(), lr=self.lr)

        # device
        self.device = get_device(device)
        print("using device: ", self.device)
        self.model.to(self.device)

    def _train_epoch(self, 
                    msg_format, 
                    teacher_forcing_ratio, 
                    clip=1.0):
        self.model.train()
        epoch_loss = 0
        train_batches_n = 0
        
        bar = tqdm(enumerate(self.datasets.train_dataloader), total=len(self.datasets.train_dataloader))
        for batch_i, (src, trg) in bar:
            if batch_i > self.max_batches_per_epoch_train:
                break
    
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Проход (Forward)
            # trg[:, :-1] - подаем всё, кроме последнего токена
            output = self.model(src, trg[:, :-1], teacher_forcing_ratio)
            
            # Подготовка данных для Loss
            # output: [batch, trg_len-1, vocab_size]
            # trg: [batch, trg_len] -> нам нужны токены со 2-го до конца
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_target = trg[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(output, trg_target)
            loss.backward()
            
            # Градиентный клиппинг (защита от взрывов градиента в LSTM)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            
            self.optimizer.step()
            epoch_loss += loss.item()
            train_batches_n += 1

            bar.set_description(msg_format.format(loss.item()))
            
        return epoch_loss / train_batches_n

    def _val(self):
        self.model.eval()
        mean_val_loss = 0
        val_batches_n = 0

        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(self.datasets.val_dataloader):
                if batch_i > self.max_batches_per_epoch_val:
                    break

                batch_x = copy_data_to_device(batch_x, self.device)
                batch_y = copy_data_to_device(batch_y, self.device)

                pred = self.model(batch_x, batch_y[:, :-1], teacher_forcing_ratio=0.0)

                output_reshape = pred.contiguous().view(-1, pred.shape[-1])
                trg = batch_y[:, 1:].contiguous().view(-1)
                loss = self.criterion(output_reshape, trg)

                mean_val_loss += loss.item()
                val_batches_n += 1

        mean_val_loss /= val_batches_n

        return mean_val_loss

    def train(self, epoch_n=None, save_after_train=True, save_epoch_model=True):
        if epoch_n is None:
            epoch_n = self.epoch_n
    
        best_val_loss = float('inf')
        mean_val_loss = float('inf')

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        best_epoch_i = 0

        for epoch in range(epoch_n):
            # train
            # Линейное снижение TF от 1.0 до 0.5 за первые 5 эпох
            if epoch < 5:
                current_tf = 1.0 - (epoch * 0.1)
            else:
                current_tf = 0.5
            mean_train_loss = self._train_epoch(f'train {epoch}/{epoch_n} -- loss: {{:3.4f}}', current_tf, clip=1.0)
            print('Среднее значение функции потерь на обучении', mean_train_loss)

            # valuation
            mean_val_loss = self._val()
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            if self.save_metrics:
                self._save_metrics(epoch, mean_train_loss, mean_val_loss)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch
                best_val_loss = mean_val_loss
                print('Новая лучшая модель!')
                if save_epoch_model:
                    self._save_model_to(self.output_path, f'{self.output_model_name}_epoch.pth')
            elif epoch - best_epoch_i > self.early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(self.early_stopping_patience))
                break

            scheduler.step(mean_val_loss)

        if save_after_train:
            self.save_model()

    def save_model(self):
        self._save_model_to(self.output_path, f'{self.output_model_name}.pth')
    
    def _save_model_to(self, path, name):
        if not exists(path):
            mkdir(path)
        
        # Собираем все артефакты
        checkpoint = {
            'model_state_dict': self.model.state_dict(), # Веса модели
            'char2id': self.datasets.get_char2id(),  # словарь[c: 1]
            'config': {                                  # Гиперпараметры для инициализации
                'emb_dim': self.model.emb_dim,
                'hidden_dim': self.model.hidden_dim,
                'n_layers': self.model.n_layers,
                'pointer_generator': True                # Пометка, что это кастомная версия
            }
        }

        # Сохраняем одним файлом
        torch.save(checkpoint, join(path, name))

    def _save_metrics(self, epoch, train_loss, val_loss):
        with open(join(self.current_path, _log_file_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])