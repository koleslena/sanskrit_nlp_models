from os.path import join, exists
from os import mkdir
import array as arr
import time

import torch
from tqdm import tqdm
import copy

from sanskrit_tagger.pos_tagger import get_device, copy_data_to_device


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
        self.best_model = model
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

        if self.optimizer_ctor is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg_alpha)
        else:
            self.optimizer = self.optimizer_ctor(self.model.parameters(), lr=self.lr)

        # device
        self.device = get_device(device)
        print("using device: ", self.device)
        model.to(self.device)

    def _train_epoch(self, msg_format):

        self.model.train()
        mean_train_loss = 0
        train_batches_n = 0
        bar = tqdm(enumerate(self.datasets.train_dataloader))
        for batch_i, (batch_x, batch_y) in bar:
            if batch_i > self.max_batches_per_epoch_train:
                break

            batch_x = copy_data_to_device(batch_x, self.device)
            batch_y = copy_data_to_device(batch_y, self.device)

            pred = self.model(batch_x)
            loss = self.criterion(pred, batch_y)

            self.model.zero_grad()
            loss.backward()

            self.optimizer.step()

            mean_train_loss += loss.item()
            train_batches_n += 1

            bar.set_description(msg_format.format(loss.item()))

        mean_train_loss /= train_batches_n

        return mean_train_loss

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

                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)

                mean_val_loss += loss.item()
                val_batches_n += 1

        mean_val_loss /= val_batches_n

        return mean_val_loss
        

    def train(self, epoch_n=None, save_after_train=True):
        if epoch_n is None:
            epoch_n = self.epoch_n

        if self.lr_scheduler_ctor is not None:
            lr_scheduler = self.lr_scheduler_ctor(self.optimizer)
        else:
            lr_scheduler = None
    
        best_epoch_i = 0
        self.best_model = copy.deepcopy(self.model)

        for epoch in range(epoch_n):
            # train
            mean_train_loss = self._train_epoch(f'train {epoch}/{epoch_n} -- loss: {{:3.4f}}')
            print('Среднее значение функции потерь на обучении', mean_train_loss)

            # valuation
            mean_val_loss = self._val()
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            if self.with_metrics:
                self._save_metrics(mean_train_loss, mean_val_loss)

            if mean_val_loss < self.best_val_loss:
                best_epoch_i = epoch
                self.best_val_loss = mean_val_loss
                self.best_model = copy.deepcopy(self.model)
                print('Новая лучшая модель!')
            elif epoch - best_epoch_i > self.early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    self.early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

        if save_after_train:
            self._save_best_model()

    def _save_best_model(self):
        if not exists(self.output_path):
            mkdir(self.output_path)
        
        torch.save(self.model.state_dict(), join(self.output_path, f'{self.output_model_name}.pth'))

        data = arr.array('i', [self.datasets.vocab_size, self.datasets.labels_num])
        with open(join(self.output_path, f'{self.output_model_name}_data.dat'), 'wb') as f:
            data.tofile(f)
        
        self.datasets.save_data(self.output_path)
    

    def _save_metrics(self, train_loss, val_loss):
        with open(join(self.current_path, f'{self.output_model_name}_metrics_train.dat'), 'a') as f:
            f.write(f"{train_loss}\n")
        with open(join(self.current_path, f'{self.output_model_name}_metrics_val.dat'), 'a') as f:
            f.write(f"{val_loss}\n")