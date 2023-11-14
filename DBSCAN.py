import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Чтение данных из таблицы features
query = """
SELECT user_id,
       username,
       email,
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

# Заменяем пропущенные значения в столбце email
data['email'] = data['email'].replace({None: ''})

# Векторизация текстовых данных (username и email)
vectorizer = CountVectorizer()
username_vectorized = vectorizer.fit_transform(data['username'])
email_vectorized = vectorizer.transform(data['email'])

# Конвертация в DataFrame и добавление к основному DataFrame
username_df = pd.DataFrame(username_vectorized.toarray(), columns=[f'username_{col}' for col in vectorizer.get_feature_names_out()])
email_df = pd.DataFrame(email_vectorized.toarray(), columns=[f'email_{col}' for col in vectorizer.get_feature_names_out()])

data = pd.concat([data, username_df, email_df], axis=1)

# Выбираем все числовые признаки для кластеризации
features_for_clustering = [
    'username_length', 'numbers_in_name', 'email_length', 'matching_names', 'pattern_email',
    'country', 'date_last_email', 'date_registered', 'date_last_login', 'matching_dates',
    'username_neighbour_above', 'username_neighbour_below', 'email_neighbour_above', 'email_neighbour_below']

# Нормализация числовых признаков
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[features_for_clustering])

# Замена null на среднее значение в каждом столбце
data_normalized = np.nan_to_num(data_normalized)

# Применение DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['cluster'] = dbscan.fit_predict(data_normalized)

# Закрываем соединение с базой данных
connection.close()

# Создание изображения
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data.index, y='username_length', hue='cluster', data=data, palette='viridis', legend='full')
plt.title('Кластеризация с помощью метода DBSCAN')
plt.xlabel('User_id')
plt.ylabel('Username length')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig('dbscan_clustering_ALL.png')
plt.show()
