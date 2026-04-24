import json
import os
from datetime import datetime

LOG_FILE = "bot_logs.json"
MAX_LOG_ENTRIES = 1000  # Храним только последние 1000 записей

def log_interaction(user_id, user_query, bot_response, latency_sec, status="success", error_message=None):
    """
    Записывает событие в лог-файл.
    Логгирует ВСЕ запросы: success, error, timeout, api_timeout, network_error
    
    Args:
        user_id: ID пользователя Telegram
        user_query: Текст запроса пользователя
        bot_response: Ответ бота
        latency_sec: Время обработки в секундах
        status: "success", "error", "timeout", "api_timeout", "network_error"
        error_message: Текст ошибки (если status != "success")
    """
    # Создаем запись
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "query_preview": user_query[:100] + ("..." if len(user_query) > 100 else ""),
        "response_length": len(bot_response) if bot_response else 0,
        "latency_sec": round(latency_sec, 2),
        "status": status
    }
    
    # Добавляем сообщение об ошибке, если есть
    if error_message:
        entry["error_message"] = error_message[:200]  # Обрезаем длинные ошибки
    
    # Читаем старый лог или создаем новый
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Ошибка чтения лога: {e}")
            logs = []
            
    # Добавляем новую запись
    logs.append(entry)
    
    # Оставляем только последние MAX_LOG_ENTRIES записей
    if len(logs) > MAX_LOG_ENTRIES:
        logs = logs[-MAX_LOG_ENTRIES:]
    
    # Сохраняем
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        # Цветной вывод в консоль
        status_emoji = {
            "success": "✅",
            "error": "❌",
            "timeout": "⏰",
            "api_timeout": "⚠️",
            "network_error": "🌐"
        }
        emoji = status_emoji.get(status, "📊")
        print(f"{emoji} Лог записан: User {user_id} | Latency {latency_sec:.2f}s | Status {status}")
        
    except Exception as e:
        print(f"❌ Ошибка записи лога: {e}")


def get_logs(user_id=None, status=None, limit=50):
    """
    Читает логи с фильтрацией.
    
    Args:
        user_id: Фильтр по ID пользователя (None = все)
        status: Фильтр по статусу (None = все)
        limit: Максимальное количество записей для возврата
    
    Returns:
        Список записей лога
    """
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except:
        return []
    
    # Фильтрация
    if user_id:
        logs = [log for log in logs if log.get('user_id') == user_id]
    if status:
        logs = [log for log in logs if log.get('status') == status]
    
    # Возвращаем последние N записей
    return logs[-limit:]


def get_stats():
    """
    Возвращает статистику по логам.
    
    Returns:
        dict с общей статистикой
    """
    logs = get_logs()
    
    if not logs:
        return {"total": 0}
    
    stats = {
        "total": len(logs),
        "success": sum(1 for log in logs if log.get('status') == 'success'),
        "error": sum(1 for log in logs if log.get('status') == 'error'),
        "timeout": sum(1 for log in logs if log.get('status') == 'timeout'),
        "avg_latency": sum(log.get('latency_sec', 0) for log in logs) / len(logs) if logs else 0
    }
    
    stats["success_rate"] = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    return stats