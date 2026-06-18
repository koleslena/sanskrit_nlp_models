# Sanskrit NLP Models 🕉️

Репозиторий содержит инструменты для обучения, проведения экспериментов и развертывания глубоких нейросетевых моделей для обработки текста на санскрите. Проект решает две ключевые задачи компьютерной лингвистики: сегментацию текста (снятие сандхи) и морфологический анализ.

---

## 🚀 Основные задачи и Архитектура моделей

Проект разделен на два независимых компонента, каждый из которых решает свою задачу:

### 1. Сегментатор текста (SanskritPointerSegmenter)
Модель для автоматического разделения слитного текста и восстановления исходных словоформ (снятие сандхи).
* **Архитектура:** Построена на базе **Seq2Seq** архитектуры.
* **Механизм внимания:** Аддитивное внимание (Additive/Bahdanau Attention).
* **Pointer-Generator Network:** Позволяет модели динамически копировать редкие или незнакомые токены напрямую из входной последовательности, что критически важно для богатой морфологии санскрита.

### 2. Морфологический классификатор (BiLSTMTagger)
Модель для предсказания грамматических признаков слов (часть речи, падеж, число, род, время и т.д.).
* **Иерархическая обработка:** 1. *Уровень букв:* Символьная BiLSTM + **Attention Pooling** извлекает морфологические признаки прямо из внутренней структуры слова, снижая проблему Out-of-Vocabulary (OOV).
  2. *Уровень предложения:* Контекстная BiLSTM анализирует синтаксические связи и разрешает омонимию.
* **Оптимизация:** Остаточные связи (Residual Connections) между уровнями букв и предложений, стабилизированные с помощью Layer Normalization.
* **Борьба с дисбалансом:** Интегрирован **Focal Loss** и взвешивание классов на основе частотности в корпусе ($1 / \sqrt{count}$).

---

## 📊 Данные и Препроцессинг

Модели обучаются на данных **Digital Corpus of Sanskrit (DCS)**. Пайплайн предобработки включает:
* Перевод в систему SLP1.
* Аугментацию редких классов на уровне предложений (умный оверсэмплинг).

---

## 🛠️ Структура проекта

```text
├── pos_taggers/
│   └── bilstm_pos_tagger.py            # Архитектура BiLSTMTagger (Char-level + Word-level BiLSTM)
├── segmenter/
│   ├── segmenter.py                    # Архитектура SanskritPointerSegmenter (Seq2Seq + Pointer)
├── common/
│   ├── loss.py                         # Кастомный Focal Loss с поддержкой маскирования паддингов
│   ├── conllu_util.py                  # Парсинг conllu файлов
│   ├── models_factory.py               # Фабрика загрузки моделей
│   └── .._datasets.py                  # Подготовка датасетов для обучения
├── train_segmenter.py                  # Скрипт запуска обучения модели сегментации
├── train_tagger.py                     # Скрипт запуска обучения морфологии
├── releases.csv                        # Релизы моделей
├── README.md
└── requirements.txt
```

## 📦 Релизы и Использование моделей

Проект ориентирован на воспроизводимость экспериментов. Обученные веса моделей, конфигурационные файлы и маппинги токенов автоматически сохраняются в Releases (Релизы) этого репозитория на GitHub.

### 📥 Загрузка через PyTorch Hub (Рекомендуемый способ)

Модели можно загружать напрямую в любой Python-скрипт без необходимости клонировать весь репозиторий, используя стандартный механизм `torch.hub`:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Загрузка морфологического классификатора (BiLSTMTagger)
tagger = torch.hub.load(
    'koleslena/sanskrit_nlp_models', 
    'pos_tagger_model', 
    version='latest',  # или конкретный тег релиза
    map_location=device
)

# 2. Загрузка сегментатора текста (SanskritPointerSegmenter)
segmenter = torch.hub.load(
    'koleslena/sanskrit_nlp_models', 
    'segmenter_model', 
    version='latest', 
    map_location=device
)
 ```


### Автоматическая загрузка моделей в код
Также возможно использовать фабрику загрузки моделей напрямую:

```Python
from common.models_factory import load_segmenter_model_from_url

# Загрузка последней стабильной версии модели на нужное устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
segmenter = load_segmenter_model_from_url(version='latest', device=device)
```

## 💻 Быстрый старт

#### Требования
- Python 3.10+
- PyTorch 2.0+

#### Установка
- Клонируйте репозиторий:
```Bash
git clone https://github.com/koleslena/sanskrit_nlp_models.git
cd sanskrit_nlp_models
```
- Установите зависимости:
```Bash
pip install -r requirements.txt
```
#### Запуск обучения (Эксперименты)
- Для запуска экспериментов с логированием параметров используйте основные скрипты:
```Bash
python train_tagger.py --epoch_n 400 --max_tokens_per_batch 2500
```
```Bash
python train_segmenter.py --train_tuning True --embedding_size 128
```
- Использование готовых скриптов `run_train_tagger.sh`, `run_train_segmenter.sh`

## 💡 Применение
#### sanskrit_tagger
Для более удобного использования моделей реализована бибилотека [sanskrit_tagger](https://github.com/koleslena/sanskrit_pos_tagger)

```bash
pip install sanskrit_tagger
```

#### Данные модели используются в качестве ядра для:
* Sanskrit Tutor Bot — Telegram-бота для интерактивного морфологического тренинга.
* Веб-интерфейсов, развернутых на платформе Hugging Face Spaces.


