import json
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from collections import defaultdict

from app.config import LLM_API_KEY
from app.models import (
    UserQuery, FormatDecision, SQLValidation, ExecutionResult, FinalResponse
)
from app.security_validator import SecurityValidator, SecurityException

client = genai.Client(api_key=LLM_API_KEY)

TABLE_SCHEMA = {
    "id": "Integer primary key",
    "transaction_id": "String transaction identifier",
    "transaction_timestamp": "Timestamp of transaction",
    "card_id": "Integer card identifier",
    "expiry_date": "String card expiry date",
    "issuer_bank_name": "String issuer bank name",
    "merchant_id": "Integer merchant identifier",
    "merchant_mcc": "Integer merchant mcc code",
    "mcc_category": "String mcc category. Only this values: (Clothing & Apparel, Dining & Restaurants, Electronics & Software, Fuel & Service Stations, General Retail & Department, Grocery & Food Markets, Hobby, Books, Sporting Goods, Home Furnishings & Supplies, Pharmacies & Health, Services (Other), Travel & Transportation, Unknown, Utilities & Bill Payments)",
    "merchant_city": "String merchant city. Example: (Astana, Almaty, Shymkent, Other)",
    "transaction_type": "String type of transaction. Example: (ATM_WITHDRAWAL, BILL_PAYMENT, ECOM, P2P_IN, P2P_OUT, POS, SALARY)",
    "transaction_amount_kzt": "Numeric amount in KZT",
    "original_amount": "Numeric original amount",
    "transaction_currency": "String currency in ISO format. Example: (ARM, BLR, CHN, GEO, ITA, KAZ, KGZ, TUR, USA, UZB)",
    "acquirer_country_iso": "String acquirer ISO. Example: ()",
    "pos_entry_mode": "String pos entry mode. Only this values: (Chip, QR_Code, Contactless, Swipe)",
    "wallet_type": "String wallet type. Only this values: (Bank's QR, Samsung Pay, Google Pay, Apple Pay)"
}

DEFAULT_LIMIT = 1000
AGGREGATION_THRESHOLD = 10000
MAX_RETRIES = 3

PRODUCTION_SYSTEM_PROMPT = f"""
SYSTEM_ROLES:
- Data Analyst Assistant
- SQL Query Optimizer  
- Security Validator
- Result Formatter

GOLDEN_RULES:
1. SCHEMA_COMPLIANCE: Строго соблюдай TABLE_SCHEMA и типы данных
2. READ_ONLY: Только SELECT запросы. Запрещены: UPDATE, DELETE, DROP, ALTER, CREATE, INSERT
3. SECURITY_FIRST: Если запрос рискован - отклони и объясни
4. CONTEXT_AWARENESS: При недостатке контекста - запроси уточнение
5. PERFORMANCE: Оптимизируй SQL (индексы, WHERE перед JOIN, LIMIT)
6. VALIDATION_LOOP: Проверяй соответствие на каждом этапе
7. TYPE_SAFETY: Все данные строго типизированы

POSTGRES_OPTIMIZATION:
- Используй EXPLAIN ANALYZE для сложных запросов
- Применяй индексные подсказки (merchant_city, transaction_timestamp)
- Ограничивай результат LIMIT {DEFAULT_LIMIT} если не указано иное
- Используй WITH для сложных агрегаций

ERROR_HANDLING:
- SQL_ERROR -> повторная генерация (макс. {MAX_RETRIES} попытки)
- NO_DATA -> понятное сообщение пользователю
- TIMEOUT -> упрощение запроса

All data in the database is stored in English. For example, city names, bank names, MCC categories, transaction types, currencies are all in English.
If the user speaks Russian or Kazakh, translate their intent to English values where appropriate. 
Always output SQL in English syntax and using English values.
Use table 'transactions'. Use lowercase column names.
Dates filtered through transaction_timestamp.
Amounts filtered through transaction_amount_kzt.
String filters must use ILIKE.

Table schema:
{json.dumps(TABLE_SCHEMA, indent=2)}
"""

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
    
    async def _determine_output_format(self, user_query: UserQuery) -> FormatDecision:
        """Определение формата вывода с учетом контекста истории"""
        history = self._get_history(user_query.user_id)
        
        context_prompt = ""
        if history:
            context_prompt = "\n\nКОНТЕКСТ ПРЕДЫДУЩИХ СООБЩЕНИЙ:\n"
            # Берем последние 3 пары сообщений для контекста
            recent_history = history[-6:] if len(history) > 6 else history
            for content in recent_history:
                role = "Пользователь" if content.role == "user" else "Ассистент"
                text = content.parts[0].text if content.parts else ""
                context_prompt += f"{role}: {text}\n"
        
        prompt = f"""
        Определи формат вывода для запроса пользователя с учетом контекста предыдущих сообщений.
        {context_prompt}
        
        ТЕКУЩИЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query.natural_language_query}
        
        Возможные форматы:
        - "text": текстовый ответ, статистика, описания
        - "table": табличные данные, списки транзакций
        - "graph": данные для графиков (временные ряды, сравнения)
        - "diagram": диаграммы, распределения
        
        ВАЖНО:
        - Пользователь может менять формат вывода в рамках одного диалога - это нормально
        - Если пользователь сначала запросил текст, а потом таблицу - это не требует уточнения
        - Требуй уточнение ТОЛЬКО если запрос действительно неясен (не указаны параметры, даты, фильтры)
        - НЕ требуй уточнение только из-за смены формата вывода
        
        Если запрос неясен или требует уточнения (например, не указаны параметры, даты, фильтры),
        верни clarification_question с уточняющим вопросом на русском языке.
        Если запрос понятен, даже если формат отличается от предыдущего, верни clarification_question: null.
        
        Верни JSON:
        {{
            "output_format": "text|table|graph|diagram",
            "confidence_score": 0.0-1.0,
            "clarification_question": null или "уточняющий вопрос на русском",
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
        
        prompt = f"""
        Проанализируй запрос пользователя и определи, достаточно ли информации для его выполнения.
        
        ЗАПРОС: {user_query.natural_language_query}
        
        Проверь:
        1. Указаны ли необходимые параметры (даты, фильтры, категории)?
        2. Понятно ли намерение пользователя?
        3. Есть ли неоднозначности в запросе?
        
        ВАЖНО:
        - Пользователь может менять формат вывода (текст/таблица/график) - это НЕ требует уточнения
        - Пользователь может задавать разные вопросы в рамках одного диалога - это нормально
        - Требуй уточнение ТОЛЬКО если запрос действительно неполон или неясен
        - НЕ требуй уточнение только из-за смены формата или типа запроса
        
        Если запрос неясен или неполон (не указаны критичные параметры), верни уточняющий вопрос на русском языке.
        Если запрос понятен, даже если он отличается от предыдущих запросов, верни is_clear: true.
        
        Верни JSON:
        {{
            "is_clear": true/false,
            "clarification_question": "уточняющий вопрос на русском или null"
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
    
    async def process_user_request(self, user_query: UserQuery) -> FinalResponse:
        """Основной пайплайн обработки запроса с поддержкой контекста"""
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
        
        # Формируем данные для промпта
        data_summary = ""
        if sql_result_data:
            # Ограничиваем количество строк для промпта (первые 20)
            preview_data = sql_result_data[:20]
            data_summary = json.dumps(preview_data, ensure_ascii=False, indent=2)
            if len(sql_result_data) > 20:
                data_summary += f"\n... и еще {len(sql_result_data) - 20} строк(и)"
        else:
            data_summary = "Нет данных"
        
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
            response = self._call_gemini(
                "Ты - помощник аналитика данных. Формируешь понятные и развернутые ответы на основе данных из базы данных.",
                prompt,
                conversation_history=history,
                use_history=True
            )
            return response.strip()
        except Exception as e:
            print(f"Error formatting text response: {e}")
            # Fallback - простое форматирование
            if sql_result_data:
                first_row = sql_result_data[0]
                values = [str(v) for v in first_row.values() if v is not None]
                return " ".join(values)
            return "Данные не найдены"
    
    def generate(self, nl_query: str) -> str:
        """Простой метод для обратной совместимости"""
        user_query = UserQuery(natural_language_query=nl_query, user_id="default")
        import asyncio
        result = asyncio.run(self.process_user_request(user_query))
        return result.metadata.get("sql_query", result.content)

def build_text2sql():
    return ProductionLLMContract()
