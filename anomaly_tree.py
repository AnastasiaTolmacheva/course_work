import sqlite3
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

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
data_imputed = data.fillna(data.mean())

# Нормализация признаков
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_imputed)

# Сортировка данных по столбцу username_length
data_sorted = data.sort_values(by='username_length')

# Вычисление матрицы расстояний
Z = linkage(data_normalized, method='single', metric='euclidean')

# Создание графика дендрограммы
plt.figure(figsize=(15, 9))
plt.title("Дендрограмма иерархической кластеризации с методом Single Linkage")
dendrogram(Z,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           labels=data.index)  # Подписываем листья как user_id

plt.savefig('anomaly_dendrogramm_single_ALL.png')
plt.show()

# Закрываем соединение с базой данных
connection.close()
