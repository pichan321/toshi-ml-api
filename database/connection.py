from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlite3

def get_database():
    SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
    # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = db()
    conn = None
    try:
        conn = sqlite3.connect('mydb.sqlite')  # Replace with your desired SQLite database file name

    except sqlite3.Error as e:
        print(e)
    
    return db