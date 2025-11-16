import json
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from collections import defaultdict

from app.config import LLM_API_KEY
from app.constants import DEFAULT_LIMIT, MAX_RETRIES, PRODUCTION_SYSTEM_PROMPT, TABLE_SCHEMA
from app.models import (
    UserQuery, FormatDecision, SQLValidation, FinalResponse
)
from app.security_validator import SecurityValidator, SecurityException

client = genai.Client(api_key=LLM_API_KEY)


class ProductionLLMContract:
    """Production-ready контракт для обработки запросов с валидацией"""
    
    def __init__(self):
        self.model = "gemini-2.5-flash"
        self.security_validator = SecurityValidator()
        self.table_schema = TABLE_SCHEMA
        # Хранилище истории диалогов по user_id
        self.conversation_history: Dict[str, List[types.Content]] = defaultdict(list)
        # Максимальное количество пар сообщений (user + model) = 10 пар = 20 Content объектов
        self.max_message_pairs = 10
    
    def _call_gemini(
        self, 
        system_instruction: str, 
        user_text: str, 
        conversation_history: Optional[List[types.Content]] = None,
        use_history: bool = True
    ) -> str:
        """Вызов Gemini API с поддержкой истории диалога"""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=5000
        )
        
        # Формируем содержимое запроса
        contents_list = []
        
        # Добавляем историю диалога если есть
        if use_history and conversation_history:
            contents_list.extend(conversation_history)
        
        # Добавляем текущий запрос пользователя
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)]
        )
        contents_list.append(user_content)
        
        response = client.models.generate_content(
            model=self.model,
            contents=contents_list,
            config=config
        )
        
        print("Gemini response received")
        return response.text
    
    def _add_to_history(self, user_id: str, user_message: str, assistant_response: str):
        """Добавление сообщений в историю диалога с автоматическим удалением старых"""
        # Добавляем сообщение пользователя
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
        self.conversation_history[user_id].append(user_content)
        
        # Добавляем ответ ассистента
        assistant_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=assistant_response)]
        )
        self.conversation_history[user_id].append(assistant_content)
        
        # Ограничиваем длину истории: храним максимум max_message_pairs пар (каждая пара = 2 Content объекта)
        max_content_objects = self.max_message_pairs * 2  # 10 пар = 20 объектов
        if len(self.conversation_history[user_id]) > max_content_objects:
            # Удаляем старые сообщения, оставляем только последние max_message_pairs пар
            self.conversation_history[user_id] = self.conversation_history[user_id][-max_content_objects:]
    
    def _get_history(self, user_id: str) -> List[types.Content]:
        """Получение истории диалога для пользователя"""
        return self.conversation_history.get(user_id, [])
    
    def _clear_history(self, user_id: str):
        """Очистка истории диалога для пользователя"""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
    
    def _detect_language(self, text: str) -> str:
        """Определение языка текста (ru, kk, en)"""
        # Простая эвристика для определения языка
        # Можно улучшить через Gemini API, но для скорости используем эвристику
        
        text_lower = text.lower()
        
        # Казахский язык - специфические символы (высокий приоритет)
        kazakh_chars = ['ә', 'ғ', 'қ', 'ң', 'ө', 'ұ', 'ү', 'һ', 'і']
        if any(char in text_lower for char in kazakh_chars):
            return "kk"
        
        # Проверка на кириллицу (русский или казахский)
        has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in text)
        
        if has_cyrillic:
            # Сначала проверяем казахские слова (специфичные)
            kazakh_words = ['қанша', 'неше', 'қайда', 'қашан', 'кім', 'не', 'бар', 'жоқ', 'саны', 'жылы', 'айы', 'транзакциялар', 'мерчанттар']
            if any(word in text_lower for word in kazakh_words):
                return "kk"
            
            # Затем проверяем русские слова (более общие)
            russian_words = ['статистика', 'транзакции', 'транзакций', 'месяц', 'месяца', 'месяцев', 'год', 'года', 'лет', 
                           'таблица', 'таблицу', 'таблицы', 'дай', 'дайте', 'покажи', 'покажите', 'разбей', 'разбейте',
                           'количество', 'сумма', 'итого', 'всего', 'мерчант', 'мерчанта', 'мерчанты', 'за', 'по', 'все']
            if any(word in text_lower for word in russian_words):
                return "ru"
            
            # Если есть кириллица, но нет специфичных слов - по умолчанию русский
            return "ru"
        
        # По умолчанию английский
        return "en"
    
    def _get_language_name(self, lang_code: str) -> str:
        """Получение названия языка для промптов"""
        lang_names = {
            "ru": "русском",
            "kk": "казахском", 
            "en": "английском"
        }
        return lang_names.get(lang_code, "русском")
    
    def _is_already_translated(self, columns: List[str]) -> bool:
        """Проверяет, переведены ли уже названия столбцов (на русский или казахский)"""
        # Проверяем наличие кириллицы в названиях столбцов
        for col in columns:
            if any('\u0400' <= char <= '\u04FF' for char in col):
                return True
        return False
    
    async def translate_column_names(
        self, 
        data: List[Dict[str, Any]], 
        user_query: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Перевод названий столбцов на язык запроса пользователя"""
        if not data:
            return data
        
        # Получаем список всех уникальных названий столбцов
        all_columns = set()
        for row in data:
            all_columns.update(row.keys())
        
        columns_list = list(all_columns)
        
        detected_lang = self._detect_language(user_query)
        print(f"Detected language for column translation: {detected_lang}, query: {user_query[:100]}")
        
        if detected_lang == "en":
            # Для английского языка перевод не нужен
            return data
        
        # Проверяем, не переведены ли уже столбцы
        if self._is_already_translated(columns_list):
            # Определяем язык текущих названий столбцов
            first_col = columns_list[0] if columns_list else ""
            has_kazakh_chars = any(char in first_col for char in ['ә', 'ғ', 'қ', 'ң', 'ө', 'ұ', 'ү', 'һ', 'і'])
            
            # Если запрос на русском, а столбцы на казахском - нужно перевести на русский
            if detected_lang == "ru" and has_kazakh_chars:
                # Переводим с казахского на русский
                # Используем обратный перевод через Gemini
                prompt = f"""
                Переведи названия столбцов с казахского языка на русский язык.
                
                КАЗАХСКИЕ НАЗВАНИЯ СТОЛБЦОВ:
                {json.dumps(columns_list, ensure_ascii=False, indent=2)}
                
                Переведи каждое название столбца с казахского на русский язык естественным и понятным образом.
                Примеры:
                - Транзакция жылы -> Год транзакции
                - Транзакция айы -> Месяц транзакции
                - Транзакциялар саны -> Количество транзакций
                - Жалпы сома (KZT) -> Общая сумма (KZT)
                - Мерчант ID -> ID мерчанта
                
                Верни JSON объект, где ключи - казахские названия, значения - русские переводы:
                {{
                    "Транзакция жылы": "Год транзакции",
                    "Транзакция айы": "Месяц транзакции",
                    ...
                }}
                """
                system_instruction = "Ты переводишь названия столбцов с казахского языка на русский язык. Давай естественные и понятные переводы."
            elif detected_lang == "kk" and not has_kazakh_chars:
                # Запрос на казахском, а столбцы на русском - переводим на казахский
                prompt = f"""
                Келесі баған атауларын орыс тілінен қазақ тіліне аудар.
                
                ОРЫС БАҒАН АТАУЛАРЫ:
                {json.dumps(columns_list, ensure_ascii=False, indent=2)}
                
                Әрбір баған атауын орыс тілінен қазақ тіліне табиғи және түсінікті түрде аудар.
                Мысалы:
                - Год транзакции -> Транзакция жылы
                - Месяц транзакции -> Транзакция айы
                - Количество транзакций -> Транзакциялар саны
                - Общая сумма (KZT) -> Жалпы сома (KZT)
                
                Верни JSON объект, где ключи - русские названия, значения - казахские переводы:
                {{
                    "Год транзакции": "Транзакция жылы",
                    "Месяц транзакции": "Транзакция айы",
                    ...
                }}
                """
                system_instruction = "Сен баған атауларын орыс тілінен қазақ тіліне аударасың. Табиғи және түсінікті аудармалар бер."
            else:
                # Язык совпадает - возвращаем как есть
                return data
            
            # Выполняем обратный перевод
            try:
                response = self._call_gemini(
                    system_instruction,
                    prompt,
                    conversation_history=None,
                    use_history=False
                )
                
                # Парсим JSON ответ
                response_clean = response.strip()
                if response_clean.startswith("```"):
                    parts = response_clean.split("```")
                    if len(parts) >= 3:
                        json_part = parts[1]
                        if json_part.startswith("json"):
                            json_part = json_part[4:].strip()
                        response_clean = json_part.strip()
                
                json_start = response_clean.find("{")
                json_end = response_clean.rfind("}")
                if json_start != -1 and json_end != -1:
                    json_str = response_clean[json_start:json_end + 1]
                    translations = json.loads(json_str)
                    
                    # Применяем переводы к данным
                    translated_data = []
                    for row in data:
                        translated_row = {}
                        for key, value in row.items():
                            translated_key = translations.get(key, key)
                            translated_row[translated_key] = value
                        translated_data.append(translated_row)
                    
                    return translated_data
            except Exception as e:
                print(f"Error re-translating column names: {e}")
                # В случае ошибки возвращаем оригинальные данные
                return data
        
        # Формируем промпт для перевода
        if detected_lang == "kk":
            prompt = f"""
            Келесі SQL сұрауының нәтижелерінен алынған баған атауларын қазақ тіліне аудар.
            
            БАҒАН АТАУЛАРЫ:
            {json.dumps(columns_list, ensure_ascii=False, indent=2)}
            
            Әрбір баған атауын қазақ тіліне табиғи және түсінікті түрде аудар.
            Мысалы:
            - transaction_count -> Транзакциялар саны
            - merchant_id -> Мерчант ID
            - total_amount -> Жалпы сома
            - avg_amount -> Орташа сома
            - transaction_amount_kzt -> Транзакция сомасы (KZT)
            - mcc_category -> MCC санаты
            - merchant_city -> Мерчант қаласы
            - transaction_year -> Транзакция жылы
            - transaction_month -> Транзакция айы
            - total_transactions -> Транзакциялар саны
            - total_amount_kzt -> Жалпы сома (KZT)
            
            КРИТИЧЕСКИ ВАЖНО: Запрос пользователя на казахском языке. Переведи ВСЕ названия столбцов на казахский язык.
            
            Верни JSON объект, где ключи - оригинальные названия, значения - переводы:
            {{
                "transaction_count": "Транзакциялар саны",
                "merchant_id": "Мерчант ID",
                ...
            }}
            """
            system_instruction = "Сен баған атауларын қазақ тіліне аударасың. Табиғи және түсінікті аудармалар бер."
        else:  # Russian
            prompt = f"""
            Переведи названия столбцов из результатов SQL запроса на русский язык.
            
            НАЗВАНИЯ СТОЛБЦОВ:
            {json.dumps(columns_list, ensure_ascii=False, indent=2)}
            
            Переведи каждое название столбца на русский язык естественным и понятным образом.
            Примеры:
            - transaction_count -> Количество транзакций
            - merchant_id -> ID мерчанта
            - total_amount -> Общая сумма
            - avg_amount -> Средняя сумма
            - transaction_amount_kzt -> Сумма транзакции (KZT)
            - mcc_category -> Категория MCC
            - merchant_city -> Город мерчанта
            - transaction_year -> Год транзакции
            - transaction_month -> Месяц транзакции
            - total_transactions -> Количество транзакций
            - total_amount_kzt -> Общая сумма (KZT)
            
            КРИТИЧЕСКИ ВАЖНО: Запрос пользователя на русском языке. Переведи ВСЕ названия столбцов на русский язык.
            НЕ используй казахский язык для переводов, даже если в истории диалога были казахские сообщения.
            
            Верни JSON объект, где ключи - оригинальные названия, значения - переводы:
            {{
                "transaction_count": "Количество транзакций",
                "merchant_id": "ID мерчанта",
                ...
            }}
            """
            system_instruction = "Ты переводишь названия столбцов на русский язык. Давай естественные и понятные переводы. НЕ используй казахский язык."
        
        try:
            # НЕ используем историю диалога при переводе столбцов, чтобы избежать влияния предыдущих языков
            response = self._call_gemini(
                system_instruction,
                prompt,
                conversation_history=None,
                use_history=False  # Не используем историю для перевода
            )
            
            # Парсим JSON ответ
            response_clean = response.strip()
            if response_clean.startswith("```"):
                parts = response_clean.split("```")
                if len(parts) >= 3:
                    json_part = parts[1]
                    if json_part.startswith("json"):
                        json_part = json_part[4:].strip()
                    response_clean = json_part.strip()
            
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = response_clean[json_start:json_end + 1]
                translations = json.loads(json_str)
                
                # Применяем переводы к данным
                translated_data = []
                for row in data:
                    translated_row = {}
                    for key, value in row.items():
                        translated_key = translations.get(key, key)
                        translated_row[translated_key] = value
                    translated_data.append(translated_row)
                
                return translated_data
        except Exception as e:
            print(f"Error translating column names: {e}")
            # В случае ошибки возвращаем оригинальные данные
            return data
        
        return data
    
    async def _determine_output_format(self, user_query: UserQuery) -> FormatDecision:
        """Определение формата вывода с учетом контекста истории"""
        history = self._get_history(user_query.user_id)
        detected_lang = self._detect_language(user_query.natural_language_query)
        lang_name = self._get_language_name(detected_lang)
        
        context_prompt = ""
        if history:
            context_prompt = "\n\nКОНТЕКСТ ПРЕДЫДУЩИХ СООБЩЕНИЙ:\n"
            # Берем последние 3 пары сообщений для контекста
            recent_history = history[-6:] if len(history) > 6 else history
            for content in recent_history:
                role = "Пользователь" if content.role == "user" else "Ассистент"
                text = content.parts[0].text if content.parts else ""
                context_prompt += f"{role}: {text}\n"
        
        # Формируем примеры в зависимости от языка
        if detected_lang == "kk":
            format_examples = """
            ПРИМЕРЫ:
            - "Қанша транзакция бар?" -> output_format: "text", clarification_question: null
            - "Транзакциялар тізімі" -> output_format: "table", clarification_question: null
            - "График көрсет" -> output_format: "graph", clarification_question: null
            """
        elif detected_lang == "en":
            format_examples = """
            EXAMPLES:
            - "How many transactions?" -> output_format: "text", clarification_question: null
            - "List transactions" -> output_format: "table", clarification_question: null
            - "Show graph" -> output_format: "graph", clarification_question: null
            """
        else:  # Russian
            format_examples = """
            ПРИМЕРЫ:
            - "Сколько транзакций?" -> output_format: "text", clarification_question: null
            - "Список транзакций" -> output_format: "table", clarification_question: null
            - "Покажи график" -> output_format: "graph", clarification_question: null
            """
        
        prompt = f"""
        Определи формат вывода для запроса пользователя с учетом контекста предыдущих сообщений.
        {context_prompt}
        
        ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query.natural_language_query}
        
        {format_examples}
        
        Возможные форматы:
        - "text": текстовый ответ, статистика, описания, вопросы "сколько", "сколько всего"
        - "table": табличные данные, списки транзакций, "покажи", "выведи список"
        - "graph": данные для графиков (временные ряды, сравнения), "график", "диаграмма"
        - "diagram": диаграммы, распределения
        
        ВАЖНО:
        - Пользователь может менять формат вывода в рамках одного диалога - это нормально
        - Если пользователь сначала запросил текст, а потом таблицу - это не требует уточнения
        - Делай умные предположения: общие вопросы о количестве = формат "text"
        - Требуй уточнение ТОЛЬКО если запрос действительно неясен и невозможно определить формат
        
        КРИТИЧЕСКИ ВАЖНО: Запрос пользователя на {lang_name} языке.
        Если нужно задать уточняющий вопрос, верни его СТРОГО на {lang_name} языке.
        
        В большинстве случаев запросы ПОНЯТНЫ и не требуют уточнения.
        Верни clarification_question: null, если можно определить формат или сделать предположение.
        Верни clarification_question ТОЛЬКО если запрос действительно неясен.
        
        Верни JSON:
        {{
            "output_format": "text|table|graph|diagram",
            "confidence_score": 0.0-1.0,
            "clarification_question": null или "уточняющий вопрос на {lang_name} языке",
            "refined_query": "уточненный запрос пользователя с учетом контекста"
        }}
        """
        
        response = self._call_gemini(
            PRODUCTION_SYSTEM_PROMPT, 
            prompt,
            conversation_history=history,
            use_history=True
        )
        try:
            # Пытаемся извлечь JSON из ответа
            response_clean = response.strip()
            if response_clean.startswith("```"):
                parts = response_clean.split("```")
                if len(parts) >= 3:
                    json_part = parts[1]
                    if json_part.startswith("json"):
                        json_part = json_part[4:].strip()
                    response_clean = json_part.strip()
            
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = response_clean[json_start:json_end + 1]
                result = json.loads(json_str)
                return FormatDecision(**result)
        except Exception as e:
            print(f"Error parsing format decision: {e}, response: {response}")
        
        # Fallback на table формат
        return FormatDecision(
            output_format="table",
            confidence_score=0.7,
            clarification_question=None,
            refined_query=user_query.natural_language_query
        )
    
    async def _load_relevant_examples(self, output_format: str, query: str) -> List:
        """Загрузка релевантных примеров (заглушка, можно расширить)"""
        return []
    
    async def _generate_and_validate_sql(
        self, 
        query: str, 
        examples: List,
        user_id: str,
        retry_count: int = 0
    ) -> SQLValidation:
        """Генерация SQL с многоуровневой валидацией и учетом контекста"""
        history = self._get_history(user_id)
        
        context_prompt = ""
        if history:
            context_prompt = "\n\nКОНТЕКСТ ПРЕДЫДУЩИХ ЗАПРОСОВ:\n"
            recent_history = history[-4:] if len(history) > 4 else history
            for content in recent_history:
                if content.role == "user":
                    text = content.parts[0].text if content.parts else ""
                    context_prompt += f"Предыдущий запрос: {text}\n"
        
        prompt = f"""
        USER_QUERY: {query}
        {context_prompt}
        SCHEMA: {json.dumps(self.table_schema, indent=2)}
        EXAMPLES: {examples}
        
        Generate optimized PostgreSQL SELECT query с учетом контекста предыдущих запросов:
        - Use indexes on merchant_city, transaction_timestamp  
        - Add WHERE conditions before JOINs
        - Include LIMIT {DEFAULT_LIMIT} if aggregating large datasets
        - Validate against user intent
        - Only SELECT queries allowed
        - Учитывай контекст предыдущих сообщений при интерпретации запроса
        
        КРИТИЧЕСКИ ВАЖНО:
        - Все названия столбцов в SQL запросе ДОЛЖНЫ быть на английском языке
        - Используй английские названия для AS алиасов: transaction_year, transaction_month, total_count, total_amount
        - НЕ используй кириллицу или казахские символы в названиях столбцов SQL
        - Примеры правильных названий: transaction_year, transaction_month, total_transactions, total_amount_kzt
        
        Return JSON:
        {{
            "sql_query": "string",
            "explanation": "string",
            "estimated_performance": "good|medium|poor"
        }}
        """
        
        try:
            response = self._call_gemini(
                PRODUCTION_SYSTEM_PROMPT, 
                prompt,
                conversation_history=history,
                use_history=True
            )
            
            # Парсим JSON ответ
            sql_query = None
            
            # Пытаемся найти JSON блок в ответе
            response_clean = response.strip()
            
            # Убираем префикс "json" если ответ начинается с него
            if response_clean.lower().startswith("json"):
                # Пропускаем слово "json" и следующий пробел/перенос строки
                lines = response_clean.split("\n")
                if lines[0].strip().lower() == "json":
                    response_clean = "\n".join(lines[1:]).strip()
                else:
                    response_clean = response_clean[4:].strip()
            
            # Убираем markdown обёртки если есть
            if response_clean.startswith("```"):
                parts = response_clean.split("```")
                if len(parts) >= 3:
                    json_part = parts[1]
                    if json_part.startswith("json"):
                        json_part = json_part[4:].strip()
                    response_clean = json_part.strip()
            
            # Пытаемся найти JSON объект в тексте
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_clean[json_start:json_end + 1]
                try:
                    result = json.loads(json_str)
                    sql_query = result.get("sql_query", None)
                    if sql_query:
                        print(f"Extracted SQL from JSON: {sql_query[:100]}...")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    pass
            
            # Если не нашли в JSON, пытаемся извлечь SQL напрямую
            if not sql_query:
                # Ищем SELECT или WITH в ответе
                sql_keywords = ["SELECT", "WITH"]
                for keyword in sql_keywords:
                    idx = response_clean.upper().find(keyword)
                    if idx != -1:
                        # Берем текст начиная с ключевого слова до конца или до следующего блока
                        potential_sql = response_clean[idx:]
                        # Убираем возможные завершающие символы после SQL
                        if ";" in potential_sql:
                            sql_query = potential_sql[:potential_sql.index(";") + 1]
                        else:
                            sql_query = potential_sql.split("\n")[0] if "\n" in potential_sql else potential_sql
                        break
            
            # Если все еще не нашли, берем весь ответ
            if not sql_query:
                sql_query = response_clean
                # Убираем префикс "json" если есть (уже обработано выше, но на всякий случай)
                if sql_query.lower().startswith("json"):
                    sql_query = sql_query[4:].strip()
            
            if not sql_query:
                raise ValueError("Could not extract SQL query from Gemini response")
            
            # Финальная очистка
            sql_query = sql_query.strip()
            if sql_query.startswith("```"):
                sql_query = sql_query.split("```")[1]
                if sql_query.startswith("sql"):
                    sql_query = sql_query[3:]
                sql_query = sql_query.strip()
            
            # Убираем точку с запятой в конце если есть (для валидации)
            sql_query_clean = sql_query.rstrip(";").strip()
            
            print(f"Final extracted SQL: {sql_query_clean[:200]}...")
            
            # Валидация безопасности
            validation = self.security_validator.validate_sql(sql_query_clean, query)
            
            # Если небезопасен и есть попытки - регенерируем
            if not validation.is_safe and retry_count < MAX_RETRIES:
                return await self._generate_and_validate_sql(query, examples, user_id, retry_count + 1)
            
            return validation
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return SQLValidation(
                sql_query="",
                is_safe=False,
                matches_intent=False,
                validation_notes=f"Ошибка генерации: {str(e)}",
                alternative_query=None
            )
    
    async def _regenerate_sql_with_feedback(self, sql_validation: SQLValidation) -> SQLValidation:
        """Регенерация SQL с учетом обратной связи"""
        # Упрощенная реализация - можно улучшить
        return sql_validation
    
    def _build_clarification_response(self, format_decision: FormatDecision, user_id: str) -> FinalResponse:
        """Построение ответа с запросом уточнения"""
        clarification_text = format_decision.clarification_question or "Требуется уточнение запроса"
        
        # Сохраняем в историю запрос уточнения
        response = FinalResponse(
            content=clarification_text,
            output_format=format_decision.output_format,
            data_preview=None,
            metadata={"requires_clarification": True}
        )
        
        # Добавляем в историю (но не сохраняем ответ ассистента, так как это уточняющий вопрос)
        # История будет обновлена когда пользователь ответит
        
        return response
    
    async def _check_query_clarity(self, user_query: UserQuery) -> Optional[str]:
        """Проверка ясности запроса и возврат уточняющего вопроса если нужно"""
        history = self._get_history(user_query.user_id)
        detected_lang = self._detect_language(user_query.natural_language_query)
        lang_name = self._get_language_name(detected_lang)
        
        # Формируем примеры умных предположений в зависимости от языка
        if detected_lang == "kk":
            examples = """
            ПРИМЕРЫ УМНЫХ ПРЕДПОЛОЖЕНИЙ:
            - "Қанша транзакция бар?" -> ПОНЯТНО: все транзакции за все время (is_clear: true)
            - "Транзакциялар саны?" -> ПОНЯТНО: все транзакции (is_clear: true)
            - "Барлық транзакциялар" -> ПОНЯТНО: все транзакции (is_clear: true)
            - "Топ мерчанттар" -> ПОНЯТНО: топ по количеству/сумме (is_clear: true)
            - "Алматыдағы транзакциялар" -> ПОНЯТНО: транзакции в Алматы (is_clear: true)
            """
        elif detected_lang == "en":
            examples = """
            EXAMPLES OF SMART ASSUMPTIONS:
            - "How many transactions?" -> CLEAR: all transactions (is_clear: true)
            - "Count transactions" -> CLEAR: all transactions (is_clear: true)
            - "All transactions" -> CLEAR: all transactions (is_clear: true)
            - "Top merchants" -> CLEAR: top by count/amount (is_clear: true)
            - "Transactions in Almaty" -> CLEAR: transactions in Almaty (is_clear: true)
            """
        else:  # Russian
            examples = """
            ПРИМЕРЫ УМНЫХ ПРЕДПОЛОЖЕНИЙ:
            - "Сколько транзакций?" -> ПОНЯТНО: все транзакции за все время (is_clear: true)
            - "Количество транзакций" -> ПОНЯТНО: все транзакции (is_clear: true)
            - "Все транзакции" -> ПОНЯТНО: все транзакции (is_clear: true)
            - "Топ мерчанты" -> ПОНЯТНО: топ по количеству/сумме (is_clear: true)
            - "Транзакции в Алматы" -> ПОНЯТНО: транзакции в Алматы (is_clear: true)
            """
        
        prompt = f"""
        Проанализируй запрос пользователя и определи, достаточно ли информации для его выполнения.
        
        ЗАПРОС: {user_query.natural_language_query}
        
        {examples}
        
        ПРАВИЛА АНАЛИЗА:
        1. Если запрос содержит общие вопросы (сколько, количество, все, топ) БЕЗ указания периода - это ПОНЯТНО, значит "за все время"
        2. Если запрос содержит фильтры (город, категория, тип) - это ПОНЯТНО, даже без даты
        3. Если намерение пользователя очевидно из контекста - это ПОНЯТНО
        4. Делай умные предположения вместо переспрашивания
        
        КОГДА ТРЕБОВАТЬ УТОЧНЕНИЕ (только в критических случаях):
        - Запрос полностью неясен или бессмыслен
        - Есть конфликтующие требования (например, "топ-10" и "все" одновременно)
        - Запрос слишком абстрактный без возможности предположения
        
        КОГДА НЕ ТРЕБОВАТЬ УТОЧНЕНИЕ:
        - Общие вопросы о количестве/сумме/топе - делай предположение "за все время"
        - Вопросы с фильтрами без даты - используй все доступные данные
        - Понятные запросы, даже если не указаны все параметры
        
        КРИТИЧЕСКИ ВАЖНО: Запрос пользователя на {lang_name} языке. 
        Если нужно задать уточняющий вопрос, верни его СТРОГО на {lang_name} языке.
        
        В большинстве случаев запросы ПОНЯТНЫ и не требуют уточнения. 
        Верни is_clear: true, если можно сделать разумное предположение.
        Верни is_clear: false ТОЛЬКО если запрос действительно неясен и невозможно предположить намерение.
        
        Верни JSON:
        {{
            "is_clear": true/false,
            "clarification_question": "уточняющий вопрос на {lang_name} языке или null"
        }}
        """
        
        try:
            response = self._call_gemini(
                PRODUCTION_SYSTEM_PROMPT,
                prompt,
                conversation_history=history,
                use_history=True
            )
            
            response_clean = response.strip()
            if response_clean.startswith("```"):
                parts = response_clean.split("```")
                if len(parts) >= 3:
                    json_part = parts[1]
                    if json_part.startswith("json"):
                        json_part = json_part[4:].strip()
                    response_clean = json_part.strip()
            
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = response_clean[json_start:json_end + 1]
                result = json.loads(json_str)
                if not result.get("is_clear", True):
                    return result.get("clarification_question")
        except Exception as e:
            print(f"Error checking query clarity: {e}")
        
        return None
    
    def _is_format_change_only(self, clarification: str) -> bool:
        """Проверяет, связан ли уточняющий вопрос только со сменой формата"""
        format_keywords = ["в виде", "в таблице", "в графике", "в диаграмме"]
        clarification_lower = clarification.lower()
        return any(keyword in clarification_lower for keyword in format_keywords)
    
    def _is_short_answer(self, query: str) -> bool:
        """Проверяет, является ли запрос коротким ответом на уточняющий вопрос"""
        short_answers = {
            "ru": ["все", "все время", "все данные", "за все время", "всего", "да", "нет"],
            "kk": ["барлық", "барлық уақыт", "барлық деректер", "барлық уақытта", "барлығы", "иә", "жоқ"],
            "en": ["all", "all time", "all data", "everything", "yes", "no"]
        }
        detected_lang = self._detect_language(query)
        query_lower = query.lower().strip()
        return query_lower in short_answers.get(detected_lang, short_answers["ru"])
    
    def _expand_short_answer(self, short_answer: str, context: str, lang: str) -> str:
        """Расширяет короткий ответ на основе контекста предыдущего вопроса"""
        # Ищем ключевые слова в контексте
        context_lower = context.lower()
        
        if lang == "kk":
            if "транзакция" in context_lower or "транзакциялар" in context_lower:
                if "қанша" in context_lower or "саны" in context_lower:
                    return "Барлық транзакциялар саны"
                return "Барлық транзакциялар"
            elif "мерчант" in context_lower:
                return "Барлық мерчанттар"
            return "Барлық деректер"
        elif lang == "en":
            if "transaction" in context_lower:
                if "how many" in context_lower or "count" in context_lower:
                    return "Count all transactions"
                return "All transactions"
            elif "merchant" in context_lower:
                return "All merchants"
            return "All data"
        else:  # Russian
            if "транзакц" in context_lower:
                if "сколько" in context_lower or "количество" in context_lower:
                    return "Количество всех транзакций"
                return "Все транзакции"
            elif "мерчант" in context_lower:
                return "Все мерчанты"
            return "Все данные"
    
    async def process_user_request(self, user_query: UserQuery) -> FinalResponse:
        """Основной пайплайн обработки запроса с поддержкой контекста"""
        # Проверяем, является ли это коротким ответом на уточняющий вопрос
        history = self._get_history(user_query.user_id)
        if self._is_short_answer(user_query.natural_language_query) and history:
            # Если это короткий ответ типа "все", расширяем его на основе контекста
            # Берем последний вопрос ассистента и предыдущий запрос пользователя из истории
            last_assistant_msg = None
            last_user_msg = None
            for content in reversed(history):
                if content.role == "model" and not last_assistant_msg:
                    last_assistant_msg = content.parts[0].text if content.parts else ""
                elif content.role == "user" and not last_user_msg:
                    last_user_msg = content.parts[0].text if content.parts else ""
                if last_assistant_msg and last_user_msg:
                    break
            
            # Если был уточняющий вопрос, расширяем короткий ответ
            if last_assistant_msg and ("уточн" in last_assistant_msg.lower() or "?" in last_assistant_msg):
                detected_lang = self._detect_language(user_query.natural_language_query)
                # Используем контекст предыдущего запроса пользователя или вопроса ассистента
                context = last_user_msg or last_assistant_msg
                expanded_query = self._expand_short_answer(
                    user_query.natural_language_query, 
                    context, 
                    detected_lang
                )
                user_query.natural_language_query = expanded_query
        
        # Шаг 0: Проверка ясности запроса
        clarification = await self._check_query_clarity(user_query)
        # Игнорируем уточняющие вопросы, связанные только со сменой формата
        if clarification and not self._is_format_change_only(clarification):
            response = FinalResponse(
                content=clarification,
                output_format="text",
                data_preview=None,
                metadata={"requires_clarification": True}
            )
            # Сохраняем запрос пользователя в историю
            self._add_to_history(user_query.user_id, user_query.natural_language_query, clarification)
            return response
        
        # Шаг 1: Определение формата с валидацией
        format_decision = await self._determine_output_format(user_query)
        
        # Игнорируем уточняющие вопросы, связанные только со сменой формата
        if format_decision.clarification_question and not self._is_format_change_only(format_decision.clarification_question):
            response = self._build_clarification_response(format_decision, user_query.user_id)
            # Сохраняем в историю
            self._add_to_history(user_query.user_id, user_query.natural_language_query, response.content)
            return response
        
        # Шаг 2: Поиск примеров и генерация SQL
        examples = await self._load_relevant_examples(
            format_decision.output_format, 
            format_decision.refined_query
        )
        
        # Шаг 3: Валидация SQL (безопасность + соответствие) с учетом истории
        sql_validation = await self._generate_and_validate_sql(
            format_decision.refined_query, 
            examples,
            user_query.user_id
        )
        
        if not sql_validation.is_safe:
            error_msg = f"Query violates security policy: {sql_validation.validation_notes}"
            self._add_to_history(user_query.user_id, user_query.natural_language_query, error_msg)
            raise SecurityException(error_msg)
            
        if not sql_validation.matches_intent:
            sql_validation = await self._regenerate_sql_with_feedback(sql_validation)
        
        # Формируем ответ с SQL
        response = FinalResponse(
            content=sql_validation.sql_query,
            output_format=format_decision.output_format,
            data_preview=None,
            metadata={
                "sql_query": sql_validation.sql_query,
                "validation_notes": sql_validation.validation_notes
            }
        )
        
        # Сохраняем в историю успешный запрос и ответ
        explanation = sql_validation.validation_notes or "SQL запрос сгенерирован успешно"
        self._add_to_history(
            user_query.user_id, 
            user_query.natural_language_query, 
            f"Сгенерирован SQL запрос: {sql_validation.sql_query[:100]}... {explanation}"
        )
        
        return response
    
    async def format_text_response(
        self, 
        user_query: str, 
        sql_result_data: List[Dict[str, Any]], 
        user_id: str
    ) -> str:
        """Генерация развернутого текстового ответа на основе результатов SQL запроса"""
        history = self._get_history(user_id)
        detected_lang = self._detect_language(user_query)
        lang_name = self._get_language_name(detected_lang)
        
        # Формируем данные для промпта
        data_summary = ""
        if sql_result_data:
            # Ограничиваем количество строк для промпта (первые 20)
            preview_data = sql_result_data[:20]
            data_summary = json.dumps(preview_data, ensure_ascii=False, indent=2)
            if len(sql_result_data) > 20:
                if detected_lang == "kk":
                    data_summary += f"\n... және тағы {len(sql_result_data) - 20} жол(дар)"
                elif detected_lang == "en":
                    data_summary += f"\n... and {len(sql_result_data) - 20} more row(s)"
                else:
                    data_summary += f"\n... и еще {len(sql_result_data) - 20} строк(и)"
        else:
            if detected_lang == "kk":
                data_summary = "Деректер жоқ"
            elif detected_lang == "en":
                data_summary = "No data"
            else:
                data_summary = "Нет данных"
        
        # Формируем промпт в зависимости от языка
        if detected_lang == "kk":
            prompt = f"""
            Сен - деректер аналитигінің көмекшісі. Пайдаланушы сұрақ қойды және SQL сұрауының нәтижелерін алды.
            
            ПАЙДАЛАНУШЫНЫҢ СҰРАҒЫ: {user_query}
            
            SQL СҰРАУЫНЫҢ НӘТИЖЕЛЕРІ:
            {data_summary}
            
            Осы деректер негізінде толық, түсінікті жауапты қазақ тілінде құрастыр.
            Жауап болуы керек:
            - Табиғи және досалым, чат-бот сияқты
            - Толық және ақпаратты
            - Құрылымдалған (қажет болса, тізімдерді пайдалануға болады)
            - Деректерден нақты сандар мен фактілерді қамтуы керек
            - Пайдаланушының сұрағына толық жауап беруі керек
            
            Егер деректер жоқ болса, мейірімділікпен хабарла.
            
            Тек жауап мәтінін қайтар, қосымша түсіндірмелер немесе метадеректерсіз.
            """
        elif detected_lang == "en":
            prompt = f"""
            You are a data analyst assistant. The user asked a question and received SQL query results.
            
            USER'S QUESTION: {user_query}
            
            SQL QUERY RESULTS:
            {data_summary}
            
            Form a detailed, clear answer in English based on this data.
            The answer should be:
            - Natural and friendly, like from a chatbot
            - Detailed and informative
            - Structured (you can use lists if appropriate)
            - Contain specific numbers and facts from the data
            - Fully answer the user's question
            
            If there is no data, politely inform about it.
            
            Return ONLY the answer text, without additional explanations or metadata.
            """
        else:  # Russian
            prompt = f"""
            Ты - помощник аналитика данных. Пользователь задал вопрос и получил результаты SQL запроса.
            
            ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}
            
            РЕЗУЛЬТАТЫ SQL ЗАПРОСА:
            {data_summary}
            
            Сформируй развернутый, понятный ответ на русском языке на основе этих данных.
            Ответ должен быть:
            - Естественным и дружелюбным, как от чат-бота
            - Развернутым и информативным
            - Структурированным (можно использовать списки, если уместно)
            - Содержать конкретные цифры и факты из данных
            - Отвечать на вопрос пользователя полностью
            
            Если данных нет, вежливо сообщи об этом.
            
            Верни ТОЛЬКО текст ответа, без дополнительных пояснений или метаданных.
            """
        
        try:
            system_instruction = "Ты - помощник аналитика данных. Формируешь понятные и развернутые ответы на основе данных из базы данных."
            if detected_lang == "kk":
                system_instruction = "Сен - деректер аналитигінің көмекшісі. Деректер базасының деректері негізінде түсінікті және толық жауаптар құрастырасың."
            elif detected_lang == "en":
                system_instruction = "You are a data analyst assistant. You form clear and detailed answers based on database data."
            
            response = self._call_gemini(
                system_instruction,
                prompt,
                conversation_history=history,
                use_history=True
            )
            return response.strip()
        except Exception as e:
            print(f"Error formatting text response: {e}")
            if sql_result_data:
                first_row = sql_result_data[0]
                values = [str(v) for v in first_row.values() if v is not None]
                result = " ".join(values)
                if detected_lang == "kk":
                    return result if result else "Деректер табылмады"
                elif detected_lang == "en":
                    return result if result else "Data not found"
                return result if result else "Данные не найдены"
            if detected_lang == "kk":
                return "Деректер табылмады"
            elif detected_lang == "en":
                return "Data not found"
            return "Данные не найдены"
    
    def generate(self, nl_query: str) -> str:
        """Простой метод для обратной совместимости"""
        user_query = UserQuery(natural_language_query=nl_query, user_id="default")
        import asyncio
        result = asyncio.run(self.process_user_request(user_query))
        return result.metadata.get("sql_query", result.content)

def build_text2sql():
    return ProductionLLMContract()
