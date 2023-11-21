import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


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
data_imputed = imputer.fit_transform(data)

# Нормализация числовых и векторизованных признаков
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_imputed)

# Применение DBSCAN
dbscan = DBSCAN(eps=1, min_samples=6)
data['cluster'] = dbscan.fit_predict(data_normalized)

# Вывод информации о фейковых аккаунтах
fake_accounts = data[data['cluster'] == -1]

# Сохранение данных в CSV файл
fake_accounts.to_csv('fake_accounts_dbscan.csv', index=True)

# Создание изображения
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data.index, y='username_length', hue='cluster', data=data, palette='viridis', legend='full')
plt.title('Кластеризация с помощью метода DBSCAN')
plt.xlabel('User_id')
plt.ylabel('Username length')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig('dbscan_clustering_ALL.png')
plt.show()

# Закрываем соединение с базой данных
connection.close()
