import sqlite3
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Чтение данных из таблицы features
query = """
SELECT user_id,
       CAST(username_length AS DOUBLE) AS username_length,
       CAST(numbers_in_name AS DOUBLE) AS numbers_in_name,
       CAST(email_length AS DOUBLE) AS email_length,
       CAST(matching_names AS DOUBLE) AS matching_names,
       CAST(pattern_email AS DOUBLE) AS pattern_email,
       CAST(country AS DOUBLE) AS country,
       CAST(date_last_email AS DOUBLE) AS date_last_email,
       CAST(matching_dates AS DOUBLE) AS matching_dates
FROM features
LIMIT 50;
"""

data = pd.read_sql_query(query, connection)

# Закрываем соединение с базой данных
connection.close()

# Выбор количества кластеров
n_clusters = 2

# Иерархическая кластеризация
model = AgglomerativeClustering(n_clusters=n_clusters)
clusters = model.fit_predict(data)

# Построение дендрограммы
to_double = model.children_.astype(np.float64)

# Вычисление матрицы расстояний
distances = pairwise_distances(data)
# Преобразование полной матрицы расстояний в сжатую форму
condensed_distances = squareform(distances)

Z = linkage(condensed_distances, method='ward')

# Создание графика дендрограммы
plt.figure(figsize=(15, 9))
plt.title("Дендрограмма иерархической кластеризации")
dendrogram(Z,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)

plt.savefig('anomaly_dendrogramm.png')
plt.show()
