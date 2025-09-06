import torch
from pathlib import Path
import os

# Конфигурационные параметры
# Параметры данных
N_SAMPLES = 1000
SEQ_LENGTH = 20
TRAIN_RATIO = 0.8

# Параметры модели
INPUT_SIZE = 1
HIDDEN_SIZE = 512
NUM_LAYERS = 4
DROPOUT = 0.3
RNN_TYPE = 'LSTM'

# Параметры обучения
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200

# Устройство
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Логирование в файл
BASE_DIR = Path(__file__).parent
RESULTS_FILENAME =  BASE_DIR / "training_results.csv"
