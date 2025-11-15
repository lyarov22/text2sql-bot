from app.sql_to_db import execute_sql_query
from app.text2sql import build_text2sql

engine = build_text2sql()

while True:
    text = input("NEW REQUEST: ")

    sql = engine.generate(text)
    print(sql)

    response = execute_sql_query(sql)

    print(response)