import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Создаем синтетические данные для акций
# X: Количество положительных новостей о компании
# Y: Изменение цены акций в процентах

# Предположим: каждая положительная новость увеличивает цену на 1.0%
# Базовая волатильность: ±0.2%

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [0.0]], dtype=torch.float32)
y_data = torch.tensor([[0.8], [1.8], [2.8], [3.8], [4.8], [-0.2]], dtype=torch.float32)

print("----------Данные для обучения----------")
print("Количество положительных новостей (X):", x_data.flatten())
print("Изменение цены акций, % (Y):", y_data.flatten())
print()

class SimplelinearModel(nn.Module):
    """
    Простейшая линейная модель (нейросеть с одним нейроном)
    Вычисляет: y = w*x + b, где:
    - w - вес (learnable parameter)
    - b - смещение (bias, learnable parameter)
    Обучается линейной зависимости y = 2x + 1
    """
    def __init__(self):
        """
        Определяем основные компоненты модели.
        Пока один линейный слой и один выход.
        """
        super(SimplelinearModel, self).__init__()
        # Создаем линейный слой: 1 входной признак -> 1 выход
        self.linear = nn.Linear(1, 1)
        print("Начальные параметры модели:")
        print(f"Вес (w): {self.linear.weight.item():.4f}")
        print(f"Смещение (b): {self.linear.bias.item():.4f}")
    
    def forward(self, x):
        """
        Определяет прямой проход данных через сеть
        Вызывается автоматически когда мы пишем model(x_data)
        """
        return self.linear(x)

# Созадем экземпляр класса
model = SimplelinearModel()

# Функция потерь - измеряет, насколько предсказания отличаются от истинных значений
criterion = nn.MSELoss()

# Оптимизатор - алгоритм, который обновляет параметры модели
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Начало обучения...")

for epoch in range(1100):
    # Подаем данные в модель и получаем предсказания
    y_prediction = model(x_data)

    # Вычисляем ошибку (loss)
    loss = criterion(y_prediction, y_data)

    # Очищаем градиент
    optimizer.zero_grad()
    # Вычисляем градиенты
    loss.backward()
    # Обновляем параметры модели на основе градиентов
    optimizer.step()

    if epoch % 100 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f'Epoch {epoch:3d} | Loss: {loss.item():.6f} | w: {w:.4f} | b: {b:.4f}')

print("Обучение завершено!")
print()

final_w = model.linear.weight.item()
final_b = model.linear.bias.item()

print("Финальные параметры модели:")
print(f"Влияние одной новости (w): {final_w:.4f} %")
print(f"Базовая волатильность (b): {final_b:.4f} %")
print(f"Ожидаемые значения: w ≈ 1.00, b ≈ -0.20")

# Визуализация
x_range = torch.linspace(0, 10, 100).reshape(-1, 1)
with torch.no_grad():
    y_range = model(x_range)

plt.figure(figsize=(12, 8))
plt.scatter(x_data.numpy(), y_data.numpy(), color='blue', 
            label='Исторические данные', s=50)
plt.plot(x_range.numpy(), y_range.numpy(), color='red', 
         linewidth=2, label='Предсказания модели')
plt.plot(x_range.numpy(), 1.0*x_range.numpy() - 0.2, color='green', 
         linestyle='--', linewidth=2, label='Идеальная линия (y=1x-0.2)') 

plt.xlabel('Количество положительных новостей')
plt.ylabel('Изменение цены акций (%)')
plt.title('Линейная регрессия: прогноз котировок акций')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()