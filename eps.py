import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler

min_samples = 6

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Чтение данных из таблицы features
query = """
SELECT user_id,
       username_length,
       numbers_in_name,
       email_length,
       matching_names,
       pattern_email,
       country,
       date_last_email,
       date_registered, 
       date_last_login,
       matching_dates,
       username_neighbour_above,
       username_neighbour_below,
       email_neighbour_above,
       email_neighbour_below
FROM features;
"""

data = pd.read_sql_query(query, connection)

# Используем user_id как индекс
data.set_index('user_id', inplace=True)

# Заполнение отсутствующих значений в данных
imputer = SimpleImputer(strategy='mean')  # Берем среднее значение по столбцу
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)

# Нормализация числовых
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_imputed)

# Обучение модели Nearest Neighbors для расчета расстояний до min_samples ближайшего соседа
neighbors_model = NearestNeighbors(n_neighbors=min_samples)
neighbors_model.fit(data_normalized)
distances, indices = neighbors_model.kneighbors()
distances = np.sort(distances[:,min_samples-1], axis=0)

# Определение точки локтя
knee_locator = KneeLocator(range(len(distances)), distances, S=1.0, curve='convex', direction='increasing')

# Построение графика
plt.plot(range(len(distances)), distances)
plt.title(f'Нахождение параметра eps для DBSCAN')
plt.xlabel(f'Points Sorted Accoarding to Distance of the {min_samples} Nearest Neighbour')
plt.ylabel(f'Distance to {min_samples}-th Nearest Neighbor')
plt.vlines(knee_locator.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color='red', label='Knee Point')

plt.savefig('dbscan_clustering_eps.png')
plt.show()

# Вывод оптимального значения eps
print("Оптимальное eps:", distances[knee_locator.knee])

# Закрываем соединение с базой данных
connection.close()
