import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Пример данных - синусоида с шумом
n_samples = 100
x_data = torch.linspace(0, 10, n_samples).reshape(-1,1)
y_data = torch.sin(x_data) + 0.1 * torch.randn(n_samples, 1)


class PerseptronModel(nn.Module):
    """
    Многослойный перцептрон для для аппроксимации зашумленных данных.

    Модель предназначена для обучения на данных с добавлением гауссовского шума
    и демонстрации способности сети выявлять основные закономерности.

    Архитектура сети:
    Вход (1 нейрон) -> Полносвязный слой (128 нейронов) -> ReLU -> Dropout -> 
    Полносвязный слой (64 нейрона) -> ReLU -> Полносвязный слой (1 нейрон) -> Выход
    """
    def __init__(self, *args, **kwargs):
        super(PerseptronModel, self).__init__()
        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        """
        Прямой проход входных данных через сеть.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

# Создание модели, функции потерь и оптимизатора
perceptron = PerseptronModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(perceptron.parameters(), lr=0.001)

print("Начало обучения...")
print()

losses = []

for epoch in range(3000):
    predictions = perceptron(x_data)
    loss = criterion(predictions, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Переключаем в режим оценки (отключаем dropout)
perceptron.eval()

# Визуализация
with torch.no_grad():
    predictions_eval = perceptron(x_data)

plt.figure(figsize=(16, 8))

# График обучения
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('График обучения')
plt.xlabel('Эпохи')
plt.ylabel('Потери')

# График предсказаний
plt.subplot(1, 2, 2)
plt.title('График предсказаний')
plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.5, label='Данные с шумом')
plt.plot(x_data.numpy(), predictions_eval.numpy(), 'r-', linewidth=2, label='Предсказания')
plt.plot(x_data.numpy(), torch.sin(x_data).numpy(), 'g--', label='Истинная синусоида')
plt.legend()

plt.tight_layout()
plt.show()

print()
print("Обучение завершено!")