import sqlite3
import pandas as pd

# Создание и соединение с базой данных
connection = sqlite3.connect('accounts_db.db')

# Создание таблицы
create_table_query = """
CREATE TABLE IF NOT EXISTS accounts (
    user_id INTEGER,
    username TEXT,
    password TEXT,
    email TEXT,
    url TEXT,
    phone TEXT,
    mailing_address TEXT,
    billing_address TEXT,
    country TEXT,
    locales TEXT,
    date_last_email TEXT,
    date_registered TEXT,
    date_validated TEXT,
    date_last_login TEXT,
    must_change_password INTEGER,
    auth_id TEXT,
    auth_str TEXT,
    disabled INTEGER,
    disabled_reason TEXT,
    inline_help TEXT,
    gossip TEXT
);
"""
connection.execute(create_table_query)

# Чтение данных из CSV файла
data_to_insert = pd.read_csv('accounts01.csv')

# Запись данных в базу данных
data_to_insert.to_sql('accounts', connection, if_exists='replace', index=False)

connection.close()
