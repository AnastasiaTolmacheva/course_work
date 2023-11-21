import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Чтение данных из базы
query = """
SELECT user_id, username_length, numbers_in_name, email_length, matching_names, pattern_email, country, date_last_email,
matching_dates, username_neighbour_above, username_neighbour_below, email_neighbour_above, email_neighbour_below
FROM features;
"""

data = pd.read_sql_query(query, connection)

# Заполнение отсутствующих значений в данных
imputer = SimpleImputer(strategy='mean')  # Берем среднее значение по столбцу
data_imputed = imputer.fit_transform(data)

# Нормализация признаков
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_imputed)

# Создание модели Isolation Forest
model = IsolationForest(contamination=0.05)  # Устанавливаем ожидаемую долю аномалий (5%)

# Обучение модели
model.fit(data_normalized)

# Предсказание аномалий
predictions = model.predict(data_normalized)

# Добавление предсказаний в исходные данные
data['anomaly'] = predictions

# Фильтрация данных с аномалиями (данные с индексом -1 - это аномалии)
fake_accounts = data[data['anomaly'] == -1]

# Сохранение данных в CSV файл
fake_accounts[['user_id']].to_csv('fake_accounts_isolation-forest.csv', index=False)

# Создание графика
plt.figure(figsize=(10, 6))
plt.scatter(data['user_id'], data['username_length'], c=data['anomaly'], cmap='viridis')
plt.xlabel('User ID')
plt.ylabel('Username length')
plt.title('Поиск аномалий методом изолированного леса')
plt.colorbar()

plt.savefig('anomaly_isolation-forest.png')
plt.show()

# Закрываем соединение с базой данных
connection.close()
