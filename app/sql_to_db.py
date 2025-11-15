import re
from sqlalchemy import create_engine, text
from app.config import DATABASE_URL

BATCH_SIZE = 100000  # Максимальный размер батча


def _is_select_query(sql_query: str) -> bool:
    """Проверяет, является ли запрос SELECT запросом."""
    return sql_query.strip().upper().startswith('SELECT')


def _add_limit_offset(sql_query: str, limit: int, offset: int) -> str:
    """
    Добавляет или модифицирует LIMIT и OFFSET в SQL запросе.
    """
    # Удаляем существующие LIMIT и OFFSET
    query_upper = sql_query.upper()
    
    # Находим позицию последнего LIMIT
    limit_match = list(re.finditer(r'\bLIMIT\s+\d+', query_upper, re.IGNORECASE))
    if limit_match:
        last_limit = limit_match[-1]
        # Удаляем LIMIT и возможный OFFSET после него
        start_pos = last_limit.start()
        # Ищем OFFSET после LIMIT
        remaining = sql_query[start_pos:]
        offset_match = re.search(r'\s+OFFSET\s+\d+', remaining, re.IGNORECASE)
        if offset_match:
            end_pos = start_pos + offset_match.end()
            sql_query = sql_query[:start_pos] + sql_query[end_pos:]
        else:
            sql_query = sql_query[:start_pos] + sql_query[last_limit.end():]
    
    # Добавляем новые LIMIT и OFFSET
    sql_query = sql_query.rstrip().rstrip(';')
    return f"{sql_query} LIMIT {limit} OFFSET {offset}"


def execute_sql_query(sql_query: str):
    """
    Выполняет SQL запрос и выводит результат в консоль.
    Оптимизировано для больших результатов: обрабатывает батчами по 100к записей.
    
    Args:
        sql_query: SQL запрос в виде строки (например, "SELECT * FROM transactions")
    """
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as connection:
        # Для не-SELECT запросов выполняем как есть
        if not _is_select_query(sql_query):
            result = connection.execute(text(sql_query))
            connection.commit()
            print("Запрос выполнен успешно.")
            return
        
        # Для SELECT запросов используем пагинацию
        offset = 0
        total_rows = 0
        columns_printed = False
        
        while True:
            # Формируем запрос с LIMIT и OFFSET
            paginated_query = _add_limit_offset(sql_query, BATCH_SIZE, offset)
            
            result = connection.execute(text(paginated_query))
            
            # Получаем колонки только один раз
            if not columns_printed:
                columns = result.keys()
                print(" | ".join(columns))
                print("-" * 80)
                columns_printed = True
            
            # Получаем батч результатов
            rows = result.fetchall()
            
            if not rows:
                break
            
            # Выводим результаты построчно
            for row in rows:
                row_values = [str(value) if value is not None else "NULL" for value in row]
                print(" | ".join(row_values))
            
            total_rows += len(rows)
            
            # Если получили меньше записей, чем запрашивали, значит это последний батч
            if len(rows) < BATCH_SIZE:
                break
            
            offset += BATCH_SIZE
            print(f"\n[Обработано {total_rows:,} строк, загружаю следующую порцию...]\n", flush=True)
        
        print(f"\n{'='*80}")
        print(f"Всего строк: {total_rows:,}")
        print(f"{'='*80}")


