from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


# Загружаем данные ирисов
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target # Указываем, что будем искать

# Показываем таблицу
print(df.head())

X = df[iris.feature_names] # матрица признаков (размеры цветков)
y = df['target'] # вектор целевых значений (виды ирисов)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Создаем модель. Испольуем алгоритм k ближайших соседей
model = KNeighborsClassifier(n_neighbors=3)
# Обучаем модель на наших данных
model.fit(X_train, y_train)
# Оцениваем точность модели на тестовых данных
accuracy = model.score(X_test, y_test)

print(f'Точность модели: {accuracy * 100:.2f}%')

# Создаем пример для предсказания
# Должно быть:
# [[5.1, 3.4, 1.4, 0.2]] - setosa
# [[6.0, 2.7, 4.0, 1.2]] - versicolor
# [[6.5, 3.0, 5.5, 2.0] - virginica
example = pd.DataFrame([[5.1, 3.4, 1.4, 0.2]] , columns=iris.feature_names)
# Предсказание
prediction = model.predict(example)

print("Предсказание вида цветка(ириса): ", iris.target_names[prediction[0]])
