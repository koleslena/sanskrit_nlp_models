import pandas as pd
import matplotlib.pyplot as plt
import os

model_release='segmenter_model_metrics_1777946740.0313993'
# Путь к твоему файлу
file_path = f'metrics/{model_release}/segmenter_train_log.csv'
save_path = f'metrics/{model_release}/learning_curves.png'

def plot_training_history(csv_path):
    # 1. Загрузка данных
    df = pd.read_csv(csv_path)
    
    # Создаем область для двух графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2. График Loss (Функция потерь)
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#6c5ce7', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#a29bfe', linestyle='--', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3. График Accuracy (Точность)
    ax2.plot(df['epoch'], df['accuracy_em'], label='Exact Match (EM)', color='#00b894', linewidth=2)
    ax2.plot(df['epoch'], df['accuracy_char'], label='Char Accuracy', color='#55efc4', linewidth=2)
    ax2.set_title('Accuracy Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Улучшаем оформление
    plt.tight_layout()
    
    # Сохранение
    plt.savefig(save_path, dpi=300)
    print(f"Графики успешно сохранены в файл: {save_path}")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(file_path):
        plot_training_history(file_path)
    else:
        print(f"Файл не найден по пути: {file_path}")