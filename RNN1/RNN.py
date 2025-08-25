import csv
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import RecurrentModel
from sklearn.preprocessing import StandardScaler


start_rnn = time.time()

def generate_sine_with_trend(n_samples):
    """Синусоида с линейным трендом"""
    x = torch.linspace(0, 4*np.pi, n_samples)
    trend = 0.03 * torch.arange(n_samples)  # Медленный рост
    y = torch.sin(x) + trend + 0.2 * torch.randn(n_samples)
    return y.unsqueeze(1)

# Генерируем данные
n_samples = 1000
data = generate_sine_with_trend(n_samples)

train_size = int(0.8 * n_samples)
train_data = data[:train_size]
test_data = data[train_size:]
test_size = n_samples - train_size

scaler = StandardScaler()
scaler.fit(train_data.numpy())

train_data_normalized = torch.FloatTensor(scaler.transform(train_data.numpy()))
test_data_normalized = torch.FloatTensor(scaler.transform(test_data.numpy()))

all_data_normalized = torch.cat([train_data_normalized, test_data_normalized], dim=0)

# Показать исходный график
# plt.figure(figsize=(12, 8))
# plt.plot(data.numpy(), label='Синусоида с линейным трендом', linewidth=2)
# plt.title('Исходные данные для обучения')
# plt.xlabel('Время')
# plt.ylabel('Значение')
# plt.legend()
# plt.grid(True)
# plt.show()

# Создаем посследовательность из нормализованных данных
def create_sequences(data, seq_length):
    """
    Создаем последовательности для обучения RNN
    """
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length] # Вход: 20 последовательных точек
        target = data[i + seq_length] # Цель: 21-я точка
        sequences.append(seq)
        targets.append(target)

    return torch.stack(sequences), torch.stack(targets)

# Длина последовательности (окно)
SEQ_LENGTH = 20

X_all, y_all = create_sequences(all_data_normalized, SEQ_LENGTH)

# Создаем последовательности из нормализованных данных
X_train = X_all[:train_size - SEQ_LENGTH]
y_train = y_all[:train_size - SEQ_LENGTH]
X_test = X_all[train_size - SEQ_LENGTH:train_size - SEQ_LENGTH + test_size]
y_test = y_all[train_size - SEQ_LENGTH:train_size - SEQ_LENGTH + test_size]

# Разделение на обучающую и тестовую выборки
print()
print(f"Обучающая выборка: {X_train.shape[0]} последовательностей")
print(f"Тестовая выборка: {X_test.shape[0]} последовательностей\n")


# Параметры модели
INPUT_SIZE = 1  # Значение временного ряда
HIDDEN_SIZE = 128 # Размер скрытого состояния
NUM_LAYERS = 2    # Количество RNN слоев
DROPOUT = 0.4    # Для предотвращения переобучения
RNN_TYPE = 'LSTM'

# Создаем модель
model = RecurrentModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, RNN_TYPE)

# Функция потерь и оптимизатор
criterion = nn.MSELoss() # Среднеквадратичная ошибка для регрессии
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print("----- Модель создана -----\n")
print("Начинаем обучение\n")

# Создаем файл для записи результатов
results_filename = "RNN1/training_results.csv"

# Создание файла для записи результатов
try:
    with open(results_filename, 'r', newline='', encoding='utf-8') as csvfile:
        pass  # Файл существует
except FileNotFoundError:
    with open(results_filename, 'w', newline='', encoding='utf-8') as csvfile:
        pass

batch_size = 32
train_losses = []
test_losses = []

for epoch in range(200):
    model.train()
    total_loss = 0

    # Обучение пакетами (batches)
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        # Прямой проход
        output, _ = model(batch_X)
        # Вычисляем потери
        loss = criterion(output, batch_y)
        loss.backward()
        # Обновляем веса
        optimizer.step()
        
        total_loss += loss.item()
    
    # Средние потери за эпоху
    avg_loss = total_loss / len(X_train)
    train_losses.append(avg_loss)

    # Оценка на тестовых данных
    model.eval()
    # Выключаем вычисление градиентов (для экономии памяти)
    with torch.no_grad():
        # 1. Делаем предсказания на ТЕСТОВЫХ данных
        test_outputs, _ = model(X_test)
        # 2. Считаем ошибку на тестовых данных
        test_loss = criterion(test_outputs, y_test).item()
        test_losses.append(test_loss)
    
    if (epoch + 1) % 20 == 0:
        epoch = f'{epoch+1}/{200}'
        with open(results_filename, 'a', newline='', encoding='utf-8') as csvfile:
            csvfile.write(f'Epoch [{epoch}], Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}\n')

        print(f'Epoch [{epoch}], Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}')

# Переключаем модель в режим оценки
model.eval()

# Делаем предсказания на всех данных
with torch.no_grad():
    # Предсказания для обучающих данных
    train_predictions, _ = model(X_train)
    
    # Предсказания для тестовых данных
    test_predictions, _ = model(X_test)

# Преобразование в исходный масштаб
train_predictions_original = scaler.inverse_transform(train_predictions.numpy())
test_predictions_original = scaler.inverse_transform(test_predictions.numpy())

# Визуализация
plt.figure(figsize=(15, 8))

# Исходные данные
plt.plot(data.numpy(), 'b-', label='Исходные данные', alpha=0.7, linewidth=2)

# Предсказания на обучающих данных
train_indices = range(SEQ_LENGTH, SEQ_LENGTH + len(train_predictions))
plt.plot(train_indices, train_predictions_original, 'g-', 
         label='Предсказания (train)', linewidth=2)

# Предсказания на тестовых данных
test_indices = range(train_size, train_size + len(test_predictions))
plt.plot(test_indices, test_predictions_original, 'r-', 
         label='Предсказания (test)', linewidth=2)

split_idx = SEQ_LENGTH + len(train_predictions)
plt.axvline(x=split_idx, color='k', linestyle='--', 
            label='Граница train/test', alpha=0.7)

plt.title('Сравнение исходных данных и предсказаний RNN')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()

print()
print("Обучение завершено!")

result_time = time.time() - start_rnn
minutes = int(result_time // 60)
seconds = int(result_time % 60)

print(f'Времени прошло: {minutes} мин {seconds} сек')