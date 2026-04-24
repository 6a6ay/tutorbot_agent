import re
from rapidfuzz import fuzz, process

# Триггеры для отчётов
REPORT_TRIGGERS = [
    r'отчёт', r'отчет', r'report',
    r'напиши.*ученик', r'расскажи.*ученик',
    r'как дела у', r'как успехи', r'успехи у',
    r'нужен отчёт', r'подготовь отчёт', r'сделай отчёт',
    r'отчёт для', r'отчёт по', r'отчет для', r'отчет по'
]

def classify_intent(text: str) -> str:
    """
    Определяет намерение пользователя.
    Returns: "report", "question", или "unknown"
    """
    text_lower = text.lower().strip()
    
    # === ЗАПРОСЫ НА ОТЧЁТ ===
    report_patterns = [
        r'отчёт', r'отчет', r'report',
        r'напиши.*ученик', r'расскажи.*ученик',
        r'как дела у', r'как успехи', r'успехи у',
        r'нужен отчёт', r'подготовь отчёт', r'сделай отчёт',
        r'отчёт для', r'отчёт по', r'отчет для', r'отчет по',
        r'про.*[а-яё]+', r'успеваемость', r'прогресс'  # "про Васю", "прогресс Маши"
    ]
    
    for pattern in report_patterns:
        if re.search(pattern, text_lower):
            return "report"
    
    # === ОБЩИЕ ВОПРОСЫ (не отчёты, но релевантные) ===
    question_patterns = [
        r'что ты умеешь', r'что ты можешь', r'твои возможности',
        r'как работать', r'как пользоваться', r'инструкция',
        r'помощь', r'help', r'команды', r'список команд'
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, text_lower):
            return "question"
    return "unknown"

def extract_name_and_class(text: str, all_students: list[dict]) -> tuple[str|None, int|None, float]:
    """
    Извлекает имя и класс из запроса с fuzzy-поиском.
    Returns: (имя, класс, confidence_score)
    """
    # Паттерн: имя + число (класс)
    match = re.search(r'([А-Яа-яЁё]+)\s*(\d{1,2})', text)
    
    if not match:
        return None, None, 0.0
    
    name_query = match.group(1).capitalize()
    try:
        class_num = int(match.group(2))
    except ValueError:
        return None, None, 0.0
    
    # Fuzzy-поиск имени в базе
    student_names = [s['name'] for s in all_students if s['class'] == class_num]
    
    if not student_names:
        return name_query, class_num, 0.0  # Вернём как есть, если класса нет в базе
    
    result = process.extractOne(name_query, student_names, scorer=fuzz.ratio)
    
    if result and result[1] >= 70:  # порог 70%
        return result[0], class_num, result[1]
    
    # Если не нашли точное совпадение, вернём исходный запрос
    return name_query, class_num, 50.0

def check_data_completeness(rag_result: dict) -> str:
    """
    Проверяет полноту данных.
    Returns: "no_data", "partial_data", или "full_data"
    """
    if not rag_result:
        return "no_data"
    
    has_topics = bool(rag_result.get('topics') and rag_result['topics'] not in ['не указаны', '', 'None'])
    has_progress = bool(rag_result.get('successes') and rag_result['successes'] not in ['не указаны', '', 'None'])
    has_difficulties = bool(rag_result.get('difficulties') and rag_result['difficulties'] not in ['не указаны', '', 'None'])
    
    if not has_topics and not has_progress and not has_difficulties:
        return "no_data"
    elif has_topics and (has_progress or has_difficulties):
        return "full_data"
    else:
        return "partial_data"
def extract_period(text: str) -> str|None:
    """
    Извлекает период (месяц) из запроса.
    Returns: название месяца или None
    """
    months = [
        'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
        'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'
    ]
    
    text_lower = text.lower()
    
    for month in months:
        # Ищем "за июль", "за июлем", "июльский", "в июле"
        if month in text_lower:
            return month.capitalize()
    
    # Проверяем на "за этот месяц", "за прошлый месяц"
    if 'этот месяц' in text_lower:
        from datetime import datetime
        return datetime.now().strftime('%B').capitalize()
    
    return None