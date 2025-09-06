import matplotlib.pyplot as plt
import numpy as np

def plot_results(data, train_predictions, test_predictions, scaler, 
                 seq_length, train_size, title='Сравнение исходных данных и предсказаний RNN'):
    """Визуализация результатов"""
    # Преобразование в исходный масштаб
    train_predictions_original = scaler.inverse_transform(train_predictions.cpu().numpy())
    test_predictions_original = scaler.inverse_transform(test_predictions.cpu().numpy())

    plt.figure(figsize=(15, 8))
    
    # Исходные данные
    plt.plot(data.numpy(), 'b-', label='Исходные данные', alpha=0.7, linewidth=2)
    
    # Предсказания
    train_indices = range(seq_length, seq_length + len(train_predictions))
    plt.plot(train_indices, train_predictions_original, 'g-', 
             label='Предсказания (train)', linewidth=2)
    
    test_indices = range(train_size, train_size + len(test_predictions))
    plt.plot(test_indices, test_predictions_original, 'r-', 
             label='Предсказания (test)', linewidth=2)
    
    split_idx = seq_length + len(train_predictions)
    plt.axvline(x=split_idx, color='k', linestyle='--', 
                label='Граница train/test', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses(train_losses, test_losses):
    """Визуализация потерь"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()