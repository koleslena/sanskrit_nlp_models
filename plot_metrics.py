import pandas as pd
import matplotlib.pyplot as plt
import os

model_release='pos_tagger_model_metrics_1778006522.3743942'

is_pos=model_release.startswith('pos')
file_name='training_log' if is_pos else 'segmenter_train_log'
# Путь к твоему файлу
file_path = f'metrics/{model_release}/{file_name}.csv'
save_path = f'metrics/{model_release}/learning_curves.png'

def plot_training_history(csv_path, is_pos):
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

    if is_pos:
        # 3. График F1 macro
        ax2.plot(df['epoch'], df['val_f1'], label='F1 score (macro)', color='#00b894', linewidth=2)
        ax2.set_title('F1 Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 score', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    else:
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
        plot_training_history(file_path, is_pos)
    else:
        print(f"Файл не найден по пути: {file_path}")