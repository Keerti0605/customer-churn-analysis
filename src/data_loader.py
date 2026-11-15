import pandas as pd
import sqlite3


def load_data():
    """
    Load data from SQLite DB
    """
    conn = sqlite3.connect("sql/churn.db")
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df