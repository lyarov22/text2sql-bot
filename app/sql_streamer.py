from sqlalchemy import create_engine, text
from sqlalchemy.engine import Result
import json
from datetime import date, datetime
from decimal import Decimal
from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    future=True,
    execution_options={"stream_results": True}  # server-side cursor
)

def json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

def stream_select_query(sql_query: str, batch_size: int = 100_000):
    """
    Энтерпрайс генератор SELECT запросов.
    Стримит JSON массивы батчами.
    Учитывает, что LIMIT может быть в SQL.
    """
    if not sql_query or not sql_query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    with engine.connect() as conn:
        result: Result = conn.execution_options(stream_results=True).execute(text(sql_query))
        columns = result.keys()
        batch = []

        for i, row in enumerate(result):
            batch.append({col: row[i] for i, col in enumerate(columns)})

            if (i + 1) % batch_size == 0:
                # отдаём батч
                yield json.dumps(batch, default=json_default) + "\n"
                batch = []
                print('отдали батч')

        # отдаём остаток
        if batch:
            yield json.dumps(batch, default=json_default) + "\n"
