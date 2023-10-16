import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Чтение данных из базы
query = """
SELECT user_id, username_length, numbers_in_name, email_length, matching_names, pattern_email, country, date_last_email, matching_dates
FROM features;
"""

data = pd.read_sql_query(query, connection)

# Создание модели Isolation Forest
model = IsolationForest(contamination=0.10)  # Устанавливаем ожидаемую долю аномалий (10%)

# Обучение модели
model.fit(data)

# Предсказание аномалий
predictions = model.predict(data)

# Добавление предсказаний в исходные данные
data['anomaly'] = predictions

# Создание графика
plt.figure(figsize=(10, 6))
plt.scatter(data['user_id'], data['username_length'], c=data['anomaly'], cmap='viridis')
plt.xlabel('User ID')
plt.ylabel('Username length')
plt.title('Поиск аномалий методом изолированного леса')
plt.colorbar()

# Сохранение графика в файл (по желанию)
plt.savefig('anomaly_detection.png')

# Отображение графика
plt.show()

# Закрываем соединение с базой данных
connection.close()
