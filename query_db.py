import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("supplychain.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Example: total quantity per material
cur.execute("""
SELECT material,
       SUM(quantity) AS total_qty
FROM orders
GROUP BY material
ORDER BY material;
""")

for row in cur.fetchall():
    print(row)

conn.close()