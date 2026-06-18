#!/bin/bash

# Установка зависимостей
pip install --upgrade pip
# Ускоряем установку, если что-то уже есть
pip install --no-cache-dir -r requirements.txt

# Выводим инфу о GPU в лог перед стартом
nvidia-smi > train_log.txt

# Запуск. Флаг -u (unbuffered) нужен, чтобы принты 
# попадали в файл моментально, а не копились в буфере.
python -u main_segmenter.py --epoch_n 200 2>&1 | tee -a train_log.txt