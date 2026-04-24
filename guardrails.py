import re

class SecurityGuardrails:
    """Защита от атак на LLM: Input/Output фильтры"""
    
    # Триггеры атак на входе
    INPUT_ATTACK_PATTERNS = [
        r'игнорируй\s+(все\s+)?инструкции',
        r'забудь\s+(все\s+)?инструкции',
        r'\bDAN\b',  # Do Anything Now
        r'(system\s+)?prompt',
        r'покажи\s+(свой\s+)?(системный\s+)?промпт',
        r'выведи\s+(JSON|базу\s+данных|metadata)',
        r'debug\s+mode',
        r'test\s+mode',
        r'распечатай\s+инструкции',
    ]
    
    # Паттерны утечки на выходе
    OUTPUT_LEAK_PATTERNS = [
        r'Ты\s+—\s+TutorBOT',
        r'SYSTEM_INSTRUCTION',
        r'ШАГ\s*\d+:',
        r'ВАЖНЫЕ\s+ПРАВИЛА:',
        r'CONSTRAINTS',
        r'OUTPUT\s+FORMAT',
    ]
    
    @classmethod
    def check_input(cls, text: str) -> tuple[bool, str]:
        """
        Проверка запроса пользователя.
        Returns: (is_safe, reason)
        """
        text_lower = text.lower()
        
        for pattern in cls.INPUT_ATTACK_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                print(f"🛡️ [INPUT GUARD] Блокировка атаки: {pattern}")
                return False, "⛔ Запрос заблокирован политикой безопасности."
        
        return True, ""
    
    @classmethod
    def check_output(cls, text: str) -> tuple[bool, str]:
        """
        Проверка ответа LLM.
        Returns: (is_safe, filtered_text)
        """
        for pattern in cls.OUTPUT_LEAK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                print(f"🛡️ [OUTPUT GUARD] Обнаружена утечка: {pattern}")
                print(f"🛡️ [OUTPUT GUARD] Заменяю ответ на заглушку")
                return False, "⛔ Запрос заблокирован политикой безопасности."
        
        return True, text