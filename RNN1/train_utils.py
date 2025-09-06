import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

def train_model(model, train_loader, X_test, y_test, criterion, optimizer, 
                num_epochs, results_filename, device):
    """Обучение модели"""
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Оценка на тестовых данных
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)

        if (epoch + 1) % 20 == 0:
            log_training_results(epoch + 1, num_epochs, avg_loss, test_loss, results_filename)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return train_losses, test_losses

def log_training_results(epoch, total_epochs, train_loss, test_loss, filename):
    """Логирование результатов обучения"""
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        csvfile.write(f'Epoch [{epoch}/{total_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}\n')
