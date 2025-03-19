import sqlite3
import pandas as pd

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Connect to SQLite database (creates a new one if it doesn't exist)
conn = sqlite3.connect("titanic.db")
cursor = conn.cursor()

# Create Titanic table
cursor.execute('''
CREATE TABLE IF NOT EXISTS titanic (
    PassengerId INTEGER PRIMARY KEY,
    Survived INTEGER,
    Pclass INTEGER,
    Name TEXT,
    Sex TEXT,
    Age REAL,
    SibSp INTEGER,
    Parch INTEGER,
    Ticket TEXT,
    Fare REAL,
    Cabin TEXT,
    Embarked TEXT
)
''')

# Insert data into SQLite
df.to_sql('titanic', conn, if_exists='replace', index=False)

# Commit and close connection
conn.commit()
conn.close()

print("Database setup complete. Titanic data is loaded.")