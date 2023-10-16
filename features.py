import sqlite3
import pandas as pd
import re
import Levenshtein

pattern = r'^\S+@\S+\.\S+$'  # Паттерн для email "user@domain.com"
threshold = 10  # Порог сходства (расстояние Левенштейна)

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
data['matching_names'] = data.apply(lambda row: int(Levenshtein.distance(row['username'], row['email'].split('@')[0]) <= threshold), axis=1)
data['pattern_email'] = data['email'].apply(lambda email: int(re.match(pattern, email) is not None))
data['country'] = data['country'].notnull().astype(int)
data['date_last_email'] = data['date_last_email'].notnull().astype(int)
data['matching_dates'] = (pd.to_datetime(data['date_last_login']) - pd.to_datetime(data['date_registered'])).dt.days <= 3

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
    matching_dates INTEGER
);
"""
connection.execute(create_features_table_query)

# Запись вычисленных характеристик в таблицу features
data[['user_id', 'username', 'username_length', 'numbers_in_name', 'email', 'email_length', 'matching_names',
      'pattern_email', 'country', 'date_last_email', 'date_registered', 'date_last_login', 'matching_dates']].to_sql('features', connection, if_exists='replace', index=False)

# Закрываем соединение с базой данных
connection.close()
