"""
Консольное приложение для преобразования текстовых запросов в SQL через Gemini API.
"""
import os
from dotenv import load_dotenv
from google import genai
from app.models import Transaction

# Загружаем переменные окружения
load_dotenv()

# Глобальная переменная для клиента (будет инициализирована в main)
client = None

def get_table_schema_prompt():
    """
    Формирует описание структуры таблицы для контекста Gemini.
    """
    schema = f"""
Таблица: {Transaction.__tablename__}

Структура таблицы transactions:
- id (Integer, PRIMARY KEY) - уникальный идентификатор транзакции
- transaction_id (String, NOT NULL) - идентификатор транзакции
- transaction_timestamp (TIMESTAMP) - время транзакции
- card_id (Integer) - идентификатор карты
- expiry_date (String) - дата истечения карты
- issuer_bank_name (String) - название банка-эмитента
- merchant_id (Integer) - идентификатор мерчанта
- merchant_mcc (Integer) - MCC код мерчанта
- mcc_category (String) - категория MCC
- merchant_city (String) - город мерчанта
- transaction_type (String) - тип транзакции
- transaction_amount_kzt (Numeric) - сумма транзакции в тенге
- original_amount (Numeric, nullable) - оригинальная сумма транзакции
- transaction_currency (String) - валюта транзакции
- acquirer_country_iso (String) - ISO код страны эквайера
- pos_entry_mode (String) - режим ввода POS
- wallet_type (String) - тип кошелька

Правила для генерации SQL:
1. Используй только таблицу 'transactions'
2. Все имена колонок должны быть в нижнем регистре
3. Используй стандартный SQL синтаксис (PostgreSQL)
4. Если пользователь запрашивает агрегацию, используй соответствующие функции (COUNT, SUM, AVG, etc.)
5. Если пользователь запрашивает фильтрацию по дате, используй transaction_timestamp
6. Если пользователь запрашивает фильтрацию по сумме, используй transaction_amount_kzt
7. Всегда возвращай только SQL запрос, без дополнительных объяснений
8. Если пользователь не указал лимит, но запрашивает список записей, добавь разумный LIMIT (например, 100)
9. Используй правильные операторы сравнения для строк (LIKE, ILIKE для поиска)
10. Для работы с датами используй функции PostgreSQL (DATE, EXTRACT, etc.)
"""
    return schema

def generate_sql(user_query: str) -> str:
    """
    Генерирует SQL запрос на основе текстового запроса пользователя.
    
    Args:
        user_query: Текстовый запрос пользователя на естественном языке
        
    Returns:
        SQL запрос в виде строки
    """
    if client is None:
        return "Ошибка: Клиент Gemini не инициализирован. Проверьте наличие API ключа."
    
    schema_prompt = get_table_schema_prompt()
    
    full_prompt = f"""{schema_prompt}

Запрос пользователя: {user_query}

Сгенерируй SQL запрос для выполнения этого запроса. Верни только SQL запрос, без дополнительных объяснений."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        sql_query = response.text.strip()
        
        # Очищаем SQL от возможных markdown форматирования
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
    
    except Exception as e:
        return f"Ошибка при генерации SQL: {str(e)}"

def main():
    """
    Основная функция для консольного взаимодействия.
    """
    global client
    
    # Инициализация клиента Gemini
    # API ключ можно задать через переменную окружения GEMINI_API_KEY или LLM_API_KEY
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("=" * 60)
        print("ОШИБКА: API ключ не найден!")
        print("=" * 60)
        print("Пожалуйста, установите переменную окружения GEMINI_API_KEY или LLM_API_KEY")
        print("в файле .env или в системных переменных окружения.")
        print("=" * 60)
        return
    
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("=" * 60)
        print("ОШИБКА: Не удалось инициализировать клиент Gemini")
        print("=" * 60)
        print(f"Детали: {str(e)}")
        print("=" * 60)
        return
    
    print("=" * 60)
    print("SQL Query Generator через Gemini API")
    print("Введите ваш запрос на естественном языке")
    print("Для выхода введите 'exit' или 'quit'")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("Ваш запрос: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("До свидания!")
                break
            
            print("\nГенерация SQL запроса...")
            sql_query = generate_sql(user_input)
            
            print("\n" + "=" * 60)
            print("Сгенерированный SQL запрос:")
            print("=" * 60)
            print(sql_query)
            print("=" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {str(e)}\n")

if __name__ == "__main__":
    main()

