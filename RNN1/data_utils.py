import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def generate_sine_with_trend(n_samples):
    """Синусоида с линейным трендом"""
    x = torch.linspace(0, 8*np.pi, n_samples)
    trend = 0.01 * torch.arange(n_samples)
    y = torch.sin(x) + trend + 0.1 * torch.randn(n_samples)
    return y.unsqueeze(1)

def create_sequences(data, seq_length):
    """Создаем последовательности для обучения RNN"""
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)

    return torch.stack(sequences), torch.stack(targets)

def prepare_data(n_samples, seq_length, train_ratio, device):
    """Подготовка и нормализация данных"""
    data = generate_sine_with_trend(n_samples)
    
    train_size = int(train_ratio * n_samples)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    scaler = StandardScaler()
    scaler.fit(train_data.numpy())
    
    train_data_normalized = torch.FloatTensor(scaler.transform(train_data.numpy()))
    test_data_normalized = torch.FloatTensor(scaler.transform(test_data.numpy()))
    
    all_data_normalized = torch.cat([train_data_normalized, test_data_normalized], dim=0)
    
    X_all, y_all = create_sequences(all_data_normalized, seq_length)
    X_all, y_all = X_all.to(device), y_all.to(device)
    
    X_train = X_all[:train_size - seq_length]
    y_train = y_all[:train_size - seq_length]
    X_test = X_all[train_size - seq_length:train_size - seq_length + len(test_data)]
    y_test = y_all[train_size - seq_length:train_size - seq_length + len(test_data)]
    
    return X_train, y_train, X_test, y_test, scaler, data

def create_data_loader(X_train, y_train, batch_size):
    """Создание DataLoader"""
    train_dataset = TensorDataset(X_train, y_train)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)