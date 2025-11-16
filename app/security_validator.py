import re
from typing import List, Set
from app.models import SQLValidation

class SecurityException(Exception):
    """Исключение для нарушений безопасности SQL"""
    pass

class SecurityValidator:
    """Валидатор безопасности SQL запросов"""
    
    DANGEROUS_PATTERNS = [
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b",
        r";\s*(\w|\s)*$",  # Multiple statements
        r"\b(COPY|GRANT|REVOKE|EXEC)\b",
        r"(\bUNION\b.*\bSELECT\b)",
        r"\b(SLEEP|BENCHMARK|WAITFOR)\b"
    ]
    
    def validate_sql(self, sql: str, user_intent: str) -> SQLValidation:
        """
        Валидация SQL запроса на безопасность и соответствие интенту
        
        Returns:
            SQLValidation с результатами валидации
        """
        sql_upper = sql.upper().strip()
        
        # Проверка синтаксической безопасности
        is_safe = True
        security_notes = []
        
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                is_safe = False
                security_notes.append(f"Обнаружен опасный паттерн: {pattern}")
        
        # Проверка что это SELECT запрос
        # if not sql_upper.startswith(("SELECT", "WITH")):
        #     is_safe = False
        #     security_notes.append("Разрешены только SELECT запросы")
        
        # Проверка соответствия интенту
        matches_intent = self._matches_intent(sql, user_intent)
        intent_notes = []
        # if not matches_intent:
        #     intent_notes.append("SQL запрос может не соответствовать намерению пользователя")
        
        validation_notes = "; ".join(security_notes + intent_notes) if (security_notes or intent_notes) else "Запрос валиден"
        
        return SQLValidation(
            sql_query=sql,
            is_safe=is_safe,
            matches_intent=matches_intent,
            validation_notes=validation_notes,
            alternative_query=None
        )
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Извлечение ключевых слов из текста"""
        # Удаляем стоп-слова и извлекаем значимые слова
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "как", "что", "где", "когда", "какой", "какая", "какие", "какое", "какую", "какого", "какой", "какую", "какие", "какое", "какого", "какой", "какую", "какие", "какое", "какого"}
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if len(w) > 2 and w not in stop_words}
    
    def _matches_intent(self, sql: str, user_intent: str) -> bool:
        """
        Проверка соответствия SQL запроса намерению пользователя
        Базовая проверка по ключевым словам
        """
        intent_keywords = self._extract_keywords(user_intent)
        sql_lower = sql.lower()
        
        # Если интент пустой или слишком общий, считаем что соответствует
        if len(intent_keywords) < 2:
            return True
        
        # Проверяем наличие ключевых слов из интента в SQL
        # (упрощенная проверка - можно улучшить с помощью LLM)
        matched_keywords = sum(1 for keyword in intent_keywords if keyword in sql_lower)
        
        # Если совпало больше половины ключевых слов, считаем что соответствует
        return matched_keywords >= len(intent_keywords) * 0.3

