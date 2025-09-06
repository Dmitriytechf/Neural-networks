import time
from config import *
from data_utils import prepare_data, create_data_loader
from train_utils import train_model, log_training_results
from visualize import plot_results, plot_losses
from models import RecurrentModel
import torch.nn as nn
import torch.optim as optim

def main():
    start_time = time.time()
    
    # Подготовка данных
    X_train, y_train, X_test, y_test, scaler, data = prepare_data(
        N_SAMPLES, SEQ_LENGTH, TRAIN_RATIO, DEVICE
    )
    
    print(f"Обучающая выборка: {X_train.shape[0]} последовательностей")
    print(f"Тестовая выборка: {X_test.shape[0]} последовательностей")
    print(f"Используемое устройство: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"Название GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("Используется CPU\n")

    
    # Создание модели
    model = RecurrentModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, RNN_TYPE)
    model = model.to(DEVICE)
    
    # Оптимизатор и функция потерь
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # DataLoader
    train_loader = create_data_loader(X_train, y_train, BATCH_SIZE)
    
    # Обучение
    train_losses, test_losses = train_model(
        model, train_loader, X_test, y_test, criterion, optimizer,
        NUM_EPOCHS, RESULTS_FILENAME, DEVICE
    )
    
    # Предсказания
    model.eval()
    with torch.no_grad():
        train_predictions, _ = model(X_train)
        test_predictions, _ = model(X_test)
    
    # Визуализация
    plot_results(data, train_predictions, test_predictions, scaler, 
                 SEQ_LENGTH, int(N_SAMPLES * TRAIN_RATIO))
    plot_losses(train_losses, test_losses)
    
    # Время выполнения
    result_time = time.time() - start_time
    minutes = int(result_time // 60)
    seconds = int(result_time % 60)
    print(f'\nВремени прошло: {minutes} мин {seconds} сек')

if __name__ == "__main__":
    main()