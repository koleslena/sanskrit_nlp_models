# Установка зависимостей
pip install -r requirements.txt

# Запуск обучения в фоновом режиме
# Все принты будут записываться в train_log.txt
nohup python main.py --data_path ./data --epoch_n 2 > train_log.txt 2>&1 &

