import sqlite3
from pathlib import Path

# DB file will sit next to this script
DB_PATH = Path(__file__).with_name("supplychain.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Simple table: orders
cur.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    material TEXT NOT NULL,
    plant TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_date TEXT NOT NULL
);
""")

# Insert a few rows
cur.executemany(
    "INSERT INTO orders (material, plant, quantity, order_date) VALUES (?, ?, ?, ?)",
    [
        ("MAT001", "PL01", 100, "2025-02-01"),
        ("MAT001", "PL01", 50, "2025-02-05"),
        ("MAT002", "PL02", 200, "2025-02-03"),
    ],
)

conn.commit()
conn.close()

print("DB created at", DB_PATH)