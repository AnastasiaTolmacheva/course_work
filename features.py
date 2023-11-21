import sqlite3
import pandas as pd
import re
import Levenshtein

pattern = r'^\S+@\S+\.\S+$'  # Паттерн для email "user@domain.com"
threshold = 6  # Порог сходства (расстояние Левенштейна)
email_similarity = 6  # Порог сходства для email

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Чтение данных из базы
query = """
SELECT user_id, username, email, country, date_last_email, date_registered, date_last_login
FROM accounts;
"""

data = pd.read_sql_query(query, connection)

# Вычисление необходимых характеристик
data['username_length'] = data['username'].str.len()
data['numbers_in_name'] = data['username'].str.contains(r'\d').astype(int)
data['email_length'] = data['email'].str.len()
data['matching_names'] = data.apply(lambda row: int(Levenshtein.distance(row['username'], row['email'].split('@')[0]) <= threshold) if row['email'] is not None else 0, axis=1)
data['pattern_email'] = data['email'].apply(lambda email: int(bool(re.match(pattern, email))) if email else 0)
data['country'] = data['country'].notnull().astype(int)
data['date_last_email'] = data['date_last_email'].notnull().astype(int)
data['date_registered'] = pd.to_datetime(data['date_registered']).astype('int64')
data['date_last_login'] = pd.to_datetime(data['date_last_login']).astype('int64')
data['matching_dates'] = (pd.to_datetime(data['date_last_login']).dt.round('S') == pd.to_datetime(data['date_registered']).dt.round('S'))
data['matching_dates'] = data['matching_dates'].astype(int)


# Функция для поиска соседей по заданному столбцу
def find_neighbours(column_name, radius):
      neighbours_above = []
      neighbours_below = []

      for i, value in enumerate(data[column_name]):
            # Поиск соседей снизу от текущей строки
            found_neighbour_below = False
            for j in range(i + 1, min(i + radius + 1, len(data))):
                  neighbour_value = data.at[j, column_name]
                  if value is not None and neighbour_value is not None and Levenshtein.distance(value,
                                                                                               neighbour_value) <= threshold:
                        neighbours_below.append(j - i)    # записываем расстояние в строках
                        found_neighbour_below = True
                        break  # прекращаем поиск после нахождения первого соседа в радиусе
            if not found_neighbour_below:
                  neighbours_below.append(None)  # None, если соседей не найдено

            # Поиск соседей сверху от текущей строки
            found_neighbour_above = False
            for j in range(i - 1, max(i - radius - 1, -1), -1):
                  neighbour_value = data.at[j, column_name]
                  if value is not None and neighbour_value is not None and Levenshtein.distance(value,
                                                                                               neighbour_value) <= threshold:
                        neighbours_above.append(i - j)  # записываем расстояние в строках
                        found_neighbour_above = True
                        break  # прекращаем поиск после нахождения первого соседа в радиусе
            if not found_neighbour_above:
                  neighbours_above.append(None)  # None, если соседей не найдено

      return neighbours_above, neighbours_below


# Определение радиуса сравнения соседей
neighbour_radius = len(data) // 3

# Добавление столбцов для соседей с похожим username и email
(data['username_neighbour_above'], data['username_neighbour_below']) = find_neighbours('username', neighbour_radius)
(data['email_neighbour_above'], data['email_neighbour_below']) = find_neighbours('email', neighbour_radius)

# Создание таблицы features
create_features_table_query = """
CREATE TABLE IF NOT EXISTS features (
    user_id INTEGER,
    username_length INTEGER,
    numbers_in_name INTEGER,
    email_length INTEGER,
    matching_names INTEGER,
    pattern_email INTEGER,
    country INTEGER,
    date_last_email INTEGER,
    date_registered INTEGER, 
    date_last_login INTEGER,
    matching_dates INTEGER,
    username_neighbour_above INTEGER,
    username_neighbour_below INTEGER,
    email_neighbour_above INTEGER,
    email_neighbour_below INTEGER
);
"""
connection.execute(create_features_table_query)

# Запись вычисленных характеристик в таблицу features
data[['user_id', 'username', 'username_length', 'numbers_in_name', 'email', 'email_length', 'matching_names',
      'pattern_email', 'country', 'date_last_email', 'date_registered', 'date_last_login', 'matching_dates',
      'username_neighbour_above', 'username_neighbour_below', 'email_neighbour_above', 'email_neighbour_below']].to_sql('features', connection, if_exists='replace', index=False)

# Закрываем соединение с базой данных
connection.close()
