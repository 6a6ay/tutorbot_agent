import os
import sys
import asyncio
import re
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler, CallbackQueryHandler
from dotenv import load_dotenv
from telegram.error import TimedOut, NetworkError
from telemetry import log_interaction

load_dotenv()

print("=" * 70)
print(" TUTORBOT TELEGRAM — ЗАПУСК")
print("=" * 70)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN не найден!")
    sys.exit(1)

print(f"✅ Токен загружен")
print("=" * 70)

from rag_agent import (
    process_query_with_rag, upsert_student_info, delete_student_info, 
    list_students,  get_student_full_profile, MONTH_ORDER, extract_name_and_class, extract_period
)

# ========== ФУНКЦИИ ВАЛИДАЦИИ ==========
def validate_input(text, expected_type="text", field_name=""):
    if not text or not text.strip():
        return False, "❌ Поле не может быть пустым. Пожалуйста, введите данные."
    clean_text = text.strip()
    if expected_type == "int":
        try:
            val = int(clean_text)
            if 1 <= val <= 11:
                return True, val
            return False, "❌ Класс должен быть числом от 1 до 11."
        except ValueError:
            return False, f"❌ Ошибка: {field_name} должен быть числом (например, 7)."
    return True, clean_text

def validate_field_content(text: str, field_name: str) -> tuple[bool, str]:
    text = text.strip()
    if len(text) < 3:
        return False, f"❌ Поле '{field_name}' слишком короткое. Опишите подробнее."
    stubs = {"ок", "ok", "да", "нет", "норм", "хорошо", "плохо", "всё", "все", "-", "—", "?"}
    if text.lower() in stubs:
        return False, f"❌ Пожалуйста, опишите '{field_name}' подробнее."
    return True, text

def is_gibberish(text: str) -> bool:
    text = text.strip()
    if not text: return True
    has_digits = any(c.isdigit() for c in text)
    has_cyrillic = any('а' <= c.lower() <= 'я' or c.lower() == 'ё' for c in text)
    if has_digits and has_cyrillic and ' ' not in text: return True
    if len(text) > 10 and ' ' not in text:
        words = re.findall(r'[а-яА-ЯёЁ]{2,}', text)
        if not words: return True
    vowels = set("аоуыэяеёюиАОУЫЭЯЕЁЮИ")
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 5:
        vowel_count = sum(1 for c in alpha_chars if c in vowels)
        if vowel_count / len(alpha_chars) < 0.2: return True
    return False

async def reject_gibberish(update: Update, text: str) -> bool:
    if is_gibberish(text):
        await update.message.reply_text("❌ Не могу разобрать введённые данные. Пожалуйста, напишите понятным текстом.", parse_mode=None)
        return True
    return False

# ========== СОСТОЯНИЯ ==========
add_state = {}
delete_state = {}
know_state = {}
edit_state = {}
profile_state = {}
report_state = {}
# pending_report_context УДАЛЁН - больше не нужен

async def post_init(application: Application) -> None:
    try:
        await application.bot.set_chat_menu_button(menu_button={"type": "commands"})
        await application.bot.set_my_commands([
            BotCommand("start", "🚀 Начать"), BotCommand("add", "➕ Добавить ученика"),
            BotCommand("profile", "👤 Просмотр профиля"), BotCommand("know", "📝 Заполнить данные"),
            BotCommand("edit", "✏️ Редактировать"), BotCommand("delete", "🗑️ Удалить"),
            BotCommand("list", "📋 Список учеников"), BotCommand("help", "❓ Помощь"),
            BotCommand("cancel", " Отменить диалог")
        ])
        print("✅ Меню и команды настроены!")
    except Exception as e:
        print(f"️ Ошибка настройки меню: {e}")

# ========== ОБРАБОТКА СООБЩЕНИЙ ==========
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = getattr(update.message, 'text', None)
    
    if not text or not text.strip():
        await update.message.reply_text(
            "❌ Я понимаю только текстовые сообщения. Пожалуйста, напишите текстом.",
            parse_mode=None
        )
        return
    
    user_id = update.message.from_user.id
    text_lower = text.lower().strip()
    
    print(f"📥 Сообщение от {user_id}: {text}")
    
    # 🛡️ ПРОВЕРКА 1: /cancel работает ВСЕГДА (даже в диалоге)
    if text_lower == "/cancel":
        await handle_cancel(update, context)
        return
    
    # 🛡️ ПРОВЕРКА 2: /start тоже сбрасывает диалог
    if text_lower == "/start":
        # Очищаем все состояния
        for state_dict in [report_state, edit_state, know_state, add_state, delete_state, profile_state]:
            if user_id in state_dict:
                del state_dict[user_id]
        
        # Запускаем start
        await handle_start(update, context)
        return

    if is_gibberish(text):
        await update.message.reply_text(
            "Простите, я вас не понимаю 😔\n"
            "Пожалуйста, задайте вопрос по поводу отчётов или используйте команды:\n"
            "/add - добавить ученика\n"
            "/know - заполнить данные\n"
            "/edit - изменить данные\n"
            "/cancel - отменить текущий диалог",
            parse_mode=None
        )
        return

    # ========== МАРШРУТИЗАЦИЯ ДИАЛОГОВ ==========
    if user_id in report_state:
        await handle_report_dialog(update, context)
        return
    if user_id in add_state:
        await handle_add_dialog(update, context)
        return
    if user_id in delete_state:
        await handle_delete_dialog(update, context)
        return
    if user_id in know_state:
        await handle_know_dialog(update, context)
        return
    if user_id in edit_state:
        await handle_edit_dialog(update, context)
        return
    if user_id in profile_state:
        await handle_profile_dialog(update, context)
        return
        
    # ========== ТЕКСТОВЫЕ ТРИГГЕРЫ ==========
    # Приоритет: сначала проверяем, не является ли это командой
    # Важно: убрали "трудности" и "успехи", чтобы они не блокировали генерацию отчёта
    
    if any(k in text_lower for k in ["добавить", "новый ученик", "записать"]):
        await handle_add_command(update, context)
        return
    if any(k in text_lower for k in ["удалить", "убрать", "стереть"]):
        await handle_delete_command(update, context)
        return
    if any(k in text_lower for k in ["список", "кого учим", "все ученики"]):
        await handle_list_command(update, context)
        return
     
    if any(k in text_lower for k in ["заполнить", "данные об ученике", "/know"]):
        await handle_know_command(update, context)
        return
        
    if any(k in text_lower for k in ["изменить", "редактировать", "править", "/edit"]):
        await handle_edit_command(update, context)
        return
        
    # ========== ИНДИКАТОР ОБРАБОТКИ И ТЕЛЕМЕТРИЯ ==========
    processing_msg = await update.message.reply_text("⏳ Идёт обработка запроса...", parse_mode=None)
    await update.message.chat.send_action(action='typing')
    
    start_time = time.time()
    status = "success"
    response = ""
    error_message = None

    try:
        print("🔍 Обработка запроса...")
        # Просто передаём сообщение агенту - он сам решит что делать
        response = await asyncio.wait_for(
            asyncio.to_thread(process_query_with_rag, text, user_id),
            timeout=30
        )
        
        if not response:
            response = "⚠️ Не удалось сгенерировать ответ. Попробуйте ещё раз."
            status = "error"
            error_message = "Empty response from RAG"

        # Просто отправляем ответ агента пользователю
        await processing_msg.delete()
        await update.message.reply_text(response, parse_mode=None)
        print("✅ Ответ отправлен")
        
    except asyncio.TimeoutError:
        await processing_msg.edit_text("⏰ Запрос обрабатывается слишком долго. Попробуйте упростить вопрос.", parse_mode=None)
        print("❌ Таймаут обработки запроса")
        status = "timeout"
        response = "⏰ Таймаут..."
    except TimedOut:
        await processing_msg.edit_text("⚠️ Произошла задержка. Попробуйте ещё раз.", parse_mode=None)
        print("❌ Таймаут Telegram API")
        status = "api_timeout"
        response = "⚠️ Задержка API..."
    except NetworkError as e:
        await processing_msg.edit_text("⚠️ Проблема с соединением.", parse_mode=None)
        print(f"❌ Ошибка сети: {e}")
        status = "network_error"
        response = "⚠️ Ошибка сети..."
    except Exception as e:
        await processing_msg.edit_text("❌ Произошла ошибка.", parse_mode=None)
        print(f"❌ Ошибка: {e}")
        status = "error"
        response = "❌ Ошибка..."
        error_message = str(e)

    finally:
        latency = time.time() - start_time
        log_interaction(
            user_id=user_id,
            user_query=text,
            bot_response=response,
            latency_sec=latency,
            status=status,
            error_message=error_message if status != "success" else None
        )

# ========== CALLBACK ОБРАБОТЧИКИ ==========
async def universal_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == 'create_report': await handle_report_command(update, context)
    elif query.data.startswith('report_period:'): await handle_report_period_callback(update, context)
    elif query.data == 'report_cancel' or query.data == 'cancel_action':
        if user_id in report_state: del report_state[user_id]
        if user_id in edit_state: del edit_state[user_id]
        if user_id in know_state: del know_state[user_id]
        if user_id in add_state: del add_state[user_id]
        if user_id in delete_state: del delete_state[user_id]
        if user_id in profile_state: del profile_state[user_id]
        await query.edit_message_text("❌ Действие отменено.")
    elif query.data == 'add_student': await handle_add_command(update, context)
    elif query.data == 'edit_student': await handle_edit_command(update, context)
    elif query.data == 'view_profile': await handle_profile_command(update, context)
    elif query.data == 'edit_view_profile':
        state = edit_state.get(user_id)
        if state and state.get('target'):
            name, class_num = state['target']['name'], state['target']['class']
            full_profile = get_student_full_profile(name, class_num)
            if full_profile:
                periods_str = ", ".join(full_profile['periods']) or "Нет данных"
                data = full_profile['data']
                def clean(v):
                    if not v: return '—'
                    c = v.replace('не указаны, ', '').replace(', не указаны', '').strip()
                    return '—' if c.lower() in ('не указаны', '') else c
                msg = f"📂 Профиль: {name} ({class_num} класс)\n Периоды: {periods_str}\n📚 Темы: {clean(data.get('topics'))}\n✅ Успехи: {clean(data.get('successes'))}\n️ Трудности: {clean(data.get('difficulties'))}"
                await query.message.reply_text(msg, parse_mode=None)
    elif query.data.startswith('edit_period:'): await handle_edit_period_callback(update, context)
    elif query.data.startswith('edit_new_period:'): await handle_edit_new_period_callback(update, context)
    elif query.data.startswith('edit_field:'): await handle_edit_field_callback(update, context)

# ========== ОТЧЁТЫ ==========
async def handle_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = update.callback_query.message if update.callback_query else update.message
    if update.callback_query: await update.callback_query.answer()
    
    if user_id in report_state:
        await msg.reply_text("Вы уже в процессе создания отчёта. Используйте /cancel.", parse_mode=None)
        return
    report_state[user_id] = {'step': 0}
    await msg.reply_text("📝 Создание отчёта\n\nВведите имя и класс ученика.\nНапример: Вася 11", parse_mode=None)

async def handle_report_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = getattr(update.message, 'text', None)
    if not text: await update.message.reply_text("❌ Отправьте текстовое сообщение.", parse_mode=None); return
    state = report_state[user_id]
    if state['step'] == 0:
        all_students = [{'name': n, 'class': c} for c, names in list_students().items() for n in names]
        name, class_num, _ = extract_name_and_class(text, all_students)
        if not name or not class_num:
            await update.message.reply_text("❌ Не удалось распознать имя и класс.\nНапишите: Вероника 10", parse_mode=None); return
        full_profile = get_student_full_profile(name, class_num)
        if not full_profile:
            await update.message.reply_text(f"😔 {name} ({class_num} класс) не найден. Добавьте через /add", parse_mode=None)
            del report_state[user_id]; return
        state.update({'name': name, 'class': class_num, 'profile': full_profile, 'step': 1})
        periods = sorted([p for p in full_profile.get('periods', []) if p.lower() != 'не указан'], key=lambda p: MONTH_ORDER.get(p.lower(), 0))
        if not periods:
            await update.message.reply_text(f" У {name} нет данных. Заполните через /know", parse_mode=None)
            del report_state[user_id]; return
        kb = [[InlineKeyboardButton(p.capitalize(), callback_data=f"report_period:{p}")] for p in periods]
        kb.append([InlineKeyboardButton("❌ Отмена", callback_data="cancel_action")])
        await update.message.reply_text(f"✅ Нашёл: {name}, {class_num} класс.\n📅 За какой период создать отчёт?", reply_markup=InlineKeyboardMarkup(kb), parse_mode=None)

async def handle_report_period_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    if user_id not in report_state: await query.edit_message_text("❌ Сессия истекла."); return
    state = report_state[user_id]
    period = query.data.split("report_period:")[1]
    name, class_num = state.get('name'), state.get('class')
    await query.edit_message_text(f"⏳ Генерирую отчёт для {name} за {period.capitalize()}...", parse_mode=None)
    try:
        response = await asyncio.wait_for(asyncio.to_thread(process_query_with_rag, f"сделай отчёт для {name} {class_num} за {period}", user_id), timeout=60)
    except asyncio.TimeoutError:
        await query.edit_message_text("⏰ Генерация заняла слишком долго."); del report_state[user_id]; return
    del report_state[user_id]
    await context.bot.send_message(chat_id=query.message.chat_id, text=response[:4000], parse_mode=None)

# ========== РЕДАКТИРОВАНИЕ ==========
async def handle_edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.callback_query.message if update.callback_query else update.message
    if update.callback_query: await update.callback_query.answer()
    edit_state[update.effective_user.id] = {'step': 0}
    await msg.reply_text("✏️ Редактирование данных\n\nВведите имя и класс (например: Вероника 10):", parse_mode=None)

async def handle_edit_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = getattr(update.message, 'text', None)
    if not text: await update.message.reply_text("❌ Отправьте текстовое сообщение.", parse_mode=None); return
    user_id = update.message.from_user.id
    state = edit_state.get(user_id)
    if not state: return
    if state['step'] in (4, 5, 6) and await reject_gibberish(update, text): return
    is_valid, result = validate_input(text, expected_type="text")
    if not is_valid: await update.message.reply_text(result, parse_mode=None); return

    if state['step'] == 0:
        parts = result.split()
        if len(parts) < 2: await update.message.reply_text("❌ Формат: Имя Класс", parse_mode=None); return
        name = parts[0].capitalize()
        is_class_valid, class_val = validate_input(parts[1], expected_type="int", field_name="Класс")
        if not is_class_valid: await update.message.reply_text(class_val, parse_mode=None); return
        full_profile = get_student_full_profile(name, class_val)
        if not full_profile: await update.message.reply_text(f"❌ {name} ({class_val} класс) не найден.", parse_mode=None); del edit_state[user_id]; return
        state.update({'target': {'name': name, 'class': class_val}, 'profile': full_profile, 'step': 1})
        await show_period_menu(update, state)
    elif state['step'] == 2:
        await update.message.reply_text("Пожалуйста, используйте кнопки выше.", parse_mode=None)
    elif state['step'] == 3:
        period_clean = text.lower().strip()
        if period_clean not in MONTH_ORDER and period_clean not in ['не указан', 'другой']:
            await update.message.reply_text(f"❌ Не распознал период '{text}'. Введите месяц.", parse_mode=None); return
        state.update({'period': period_clean, 'step': 4})
        await update.message.reply_text(f" Период: {period_clean.capitalize()}\n\nВведите темы занятий (через запятую):", parse_mode=None)
    elif state['step'] == 4:
        if state.get('is_new_period'): await _save_new_period(update, state, result)
        else: await _save_existing_field(update, state, result)
    elif state['step'] == 5:
        await _save_new_period_continued(update, state, result)
    elif state['step'] == 6:
        await _save_new_period_final(update, state, result)

async def show_period_menu(update: Update, state):
    name, class_num = state['target']['name'], state['target']['class']
    periods = sorted([p for p in state.get('profile', {}).get('periods', []) if p.lower() != 'не указан'], key=lambda p: MONTH_ORDER.get(p.lower(), 0))
    kb = [[InlineKeyboardButton(f"✏️ Редактировать {p.capitalize()}", callback_data=f"edit_period:{name}:{class_num}:{p}")] for p in periods]
    kb.append([InlineKeyboardButton("➕ Добавить новый период", callback_data=f"edit_new_period:{name}:{class_num}")])
    kb.append([InlineKeyboardButton("❌ Отмена", callback_data="cancel_action")])
    await update.message.reply_text(f" {name}, {class_num} класс\n Периоды: {', '.join(p.capitalize() for p in periods) or 'нет'}\n\nЧто делаем?", reply_markup=InlineKeyboardMarkup(kb), parse_mode=None)

async def handle_edit_period_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    user_id = query.from_user.id; data = query.data
    if not data.startswith("edit_period:"): return
    _, name, class_num, period = data.split(":", 3)
    class_num = int(class_num)
    edit_state[user_id] = {'step': 2, 'target': {'name': name, 'class': class_num}, 'period': period, 'is_new_period': False}
    kb = [[InlineKeyboardButton("👤 Просмотр профиля", callback_data="edit_view_profile")],
          [InlineKeyboardButton(" Темы", callback_data="edit_field:topics")],
          [InlineKeyboardButton("✅ Успехи", callback_data="edit_field:successes")],
          [InlineKeyboardButton("⚠️ Трудности", callback_data="edit_field:difficulties")],
          [InlineKeyboardButton("❌ Отмена", callback_data="cancel_action")]]
    await query.edit_message_text(f"✏️ Редактирование: {name}, {class_num} класс, {period.capitalize()}\n\nЧто редактируем?", reply_markup=InlineKeyboardMarkup(kb))

async def handle_edit_new_period_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    user_id = query.from_user.id; data = query.data
    if not data.startswith("edit_new_period:"): return
    _, name, class_num = data.split(":", 2); class_num = int(class_num)
    edit_state[user_id] = {'step': 3, 'target': {'name': name, 'class': class_num}, 'is_new_period': True}
    await query.edit_message_text(f" Новый период для {name}, {class_num} класс\n\nВведите название периода:")

async def handle_edit_field_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    user_id = query.from_user.id; state = edit_state.get(user_id)
    if not query.data.startswith("edit_field:") or not state: return
    field = query.data.split("edit_field:")[1]
    state.update({'editing_field': field, 'step': 4})
    names = {'topics': 'темы занятий', 'successes': 'успехи', 'difficulties': 'трудности'}
    await query.edit_message_text(f"✏️ Введите {names.get(field, field)} для {state['target']['name']} ({state.get('period', 'новый период').capitalize()}).\nНовые данные будут ДОБАВЛЕНЫ.", parse_mode=None)

async def _save_new_period(update: Update, state, topics):
    edit_state[update.message.from_user.id].update({'step': 5, 'topics': topics})
    await update.message.reply_text("✅ Темы сохранены.\n\nВведите успехи ученика:", parse_mode=None)
async def _save_new_period_continued(update: Update, state, successes):
    edit_state[update.message.from_user.id].update({'step': 6, 'successes': successes})
    await update.message.reply_text("✅ Успехи сохранены.\n\nВведите трудности ученика:", parse_mode=None)
async def _save_new_period_final(update: Update, state, difficulties):
    name, class_num, period = state['target']['name'], state['target']['class'], state['period']
    try:
        upsert_student_info(name, class_num, period, state.get('topics','не указаны'), state.get('successes','не указаны'), difficulties, merge_mode=False)
        del edit_state[update.message.from_user.id]
        await update.message.reply_text(f"✅ Добавлен период {period.capitalize()} для {name}!", parse_mode=None)
    except ValueError as e: await update.message.reply_text(f"🛡️ Ошибка безопасности: {e}")
    except Exception: await update.message.reply_text("❌ Ошибка сохранения.")

async def _save_existing_field(update: Update, state, new_value):
    name, class_num, period, field = state['target']['name'], state['target']['class'], state['period'], state.get('editing_field')
    profile = get_student_full_profile(name, class_num, period)
    if not profile or not profile.get('period_found'): await update.message.reply_text("❌ Данные не найдены."); del edit_state[update.message.from_user.id]; return
    current = profile['data']
    updated = {k: current.get(k, 'не указаны') for k in ['topics', 'successes', 'difficulties']}
    updated[field] = new_value
    try:
        upsert_student_info(name, class_num, period, **updated, merge_mode=True)
        del edit_state[update.message.from_user.id]
        await update.message.reply_text(f"✅ Поле обновлено!", parse_mode=None)
    except ValueError as e: await update.message.reply_text(f"️ Ошибка: {e}")
    except Exception: await update.message.reply_text("❌ Ошибка сохранения.")

# ========== /KNOW ==========
async def handle_know_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = getattr(update.message, 'text', None)
    if not text: await update.message.reply_text("❌ Отправьте текстовое сообщение.", parse_mode=None); return
    user_id = update.message.from_user.id
    state = know_state[user_id]
    step = state['step']
    if step in (2, 3, 4) and await reject_gibberish(update, text): return
    is_valid, result = validate_input(text, expected_type="text")
    if not is_valid: await update.message.reply_text(result, parse_mode=None); return

    if step == 0:
        parts = result.split()
        if len(parts) < 2: await update.message.reply_text("❌ Формат: Имя Класс", parse_mode=None); return
        if not parts[1].isdigit(): await update.message.reply_text("❌ Класс должен быть числом.", parse_mode=None); return
        state['data']['name'] = parts[0].capitalize()
        _, class_val = validate_input(parts[1], expected_type="int")
        state['data']['class'] = class_val
        state['step'] = 1
        await update.message.reply_text("📅 За какой период? (например: ноябрь):", parse_mode=None)
    elif step == 1:
        state['data']['period'] = result.lower().strip()
        existing = get_student_full_profile(state['data']['name'], state['data']['class'], requested_period=state['data']['period'])
        state['merge_mode'] = bool(existing and existing.get('period_found'))
        state['step'] = 2
        await update.message.reply_text(" Какие темы проходили? (через запятую):", parse_mode=None)
    elif step == 2:
        is_ok, val = validate_field_content(result, "темы занятий")
        if not is_ok: await update.message.reply_text(val, parse_mode=None); return
        state['data']['topics'] = val
        state['step'] = 3
        await update.message.reply_text("✅ Какие успехи/достижения?", parse_mode=None)
    elif step == 3:
        is_ok, val = validate_field_content(result, "успехи")
        if not is_ok: await update.message.reply_text(val, parse_mode=None); return
        state['data']['successes'] = val
        state['step'] = 4
        await update.message.reply_text("✅ Какие трудности?", parse_mode=None)
    elif step == 4:
        is_ok, val = validate_field_content(result, "трудности")
        if not is_ok: await update.message.reply_text(val, parse_mode=None); return
        state['data']['difficulties'] = val
        try:
            upsert_student_info(**state['data'], merge_mode=state.get('merge_mode', False))
            await update.message.reply_text(f"✅ Данные о {state['data']['name']} за {state['data']['period']} сохранены!", parse_mode=None)
            del know_state[user_id]
        except ValueError as e:
            await update.message.reply_text(f"⛔ Ошибка безопасности: {e}")
            state['step'] = 2
            await update.message.reply_text("📚 Введите темы заново:", parse_mode=None)
        except Exception:
            await update.message.reply_text("❌ Ошибка сохранения.")

async def handle_know_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    know_state[user_id] = {'step': 0, 'data': {}}
    await update.message.reply_text("📝 Заполнение данных\n\nВведите имя и класс (например: Вероника 10):", parse_mode=None)

# ========== /ADD ==========
async def handle_add_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = getattr(update.message, 'text', None)
    if not text: await update.message.reply_text("❌ Отправьте текстовое сообщение.", parse_mode=None); return
    user_id = update.message.from_user.id
    state = add_state[user_id]
    step = state['step'] # ✅ ИСПРАВЛЕНО: step теперь определён до использования
    if step in (0, 2, 3, 4) and await reject_gibberish(update, text): return
    is_valid, result = validate_input(text, expected_type="text")
    if not is_valid: await update.message.reply_text(result, parse_mode=None); return
    if step == 0:
        state['data']['name'] = result.capitalize()
        state['step'] = 1
        await update.message.reply_text("2️⃣ Класс (число от 1 до 11):", parse_mode=None)
    else:
        is_class_valid, class_val = validate_input(result, expected_type="int", field_name="Класс")
        if not is_class_valid: await update.message.reply_text(class_val, parse_mode=None); return
        state['data']['class'] = class_val
        upsert_student_info(state['data']['name'], class_val, "не указан", "не указаны", "не указаны", "не указаны", merge_mode=False)
        await update.message.reply_text(f"✅ {state['data']['name']} ({class_val} класс) добавлен!", parse_mode=None)
        del add_state[user_id]

async def handle_add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.callback_query.from_user.id if update.callback_query else update.message.from_user.id
    if update.callback_query: await update.callback_query.answer()
    add_state[user_id] = {'step': 0, 'data': {}}
    msg = update.callback_query.message if update.callback_query else update.message
    await msg.reply_text("➕ Добавление ученика\n\nВведите имя:", parse_mode=None)

# ========== /DELETE & /LIST & /START & /HELP & /CANCEL & /PROFILE ==========
async def handle_delete_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = getattr(update.message, 'text', None)
    if not text: await update.message.reply_text("❌ Отправьте текстовое сообщение.", parse_mode=None); return
    user_id = update.message.from_user.id
    is_valid, result = validate_input(text, expected_type="text")
    if not is_valid: await update.message.reply_text(result, parse_mode=None); return
    try:
        parts = result.split()
        if len(parts) < 2: await update.message.reply_text("❌ Формат: Имя Класс", parse_mode=None); return
        name = parts[0].capitalize()
        _, class_val = validate_input(parts[1], expected_type="int", field_name="Класс")
        if delete_student_info(name, class_val): await update.message.reply_text(f"🗑️ {name} ({class_val} класс) удалён.", parse_mode=None)
        else: await update.message.reply_text("❌ Ученик не найден.", parse_mode=None)
    except Exception: await update.message.reply_text("❌ Ошибка удаления.")
    finally: delete_state.pop(user_id, None)

async def handle_delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    delete_state[update.message.from_user.id] = True
    await update.message.reply_text("🗑️ Удаление ученика\n\nИмя Класс:", parse_mode=None)

async def handle_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    students = list_students()
    if not students: await update.message.reply_text("📋 Список пуст", parse_mode=None); return
    msg = "📋 СПИСОК УЧЕНИКОВ\n\n"
    for c in sorted(students.keys()): msg += f"{c} класс: {', '.join(students[c])}\n"
    await update.message.reply_text(msg, parse_mode=None)

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("📝 Создать отчёт", callback_data='create_report')],
          [InlineKeyboardButton("➕ Добавить", callback_data='add_student'), InlineKeyboardButton("✏️ Редактировать", callback_data='edit_student')],
          [InlineKeyboardButton("👤 Профиль", callback_data='view_profile'), InlineKeyboardButton("❌ Отмена", callback_data='cancel_action')]]
    await update.message.reply_text(
    "🌸 Привет! Я TutorBOT.\n\n"
    "Команды:\n"
    "/add - добавить\n"
    "/know - заполнить\n"
    "/edit - изменить\n"
    "/profile - посмотреть профиль\n"
    "/list - список",
    reply_markup=InlineKeyboardMarkup(kb),
    parse_mode=None
)

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE): await update.message.reply_text("🆘 Поддержка: @workflow23", parse_mode=None)

async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена текущего диалога"""
    user_id = update.message.from_user.id
    
    for state_dict in [add_state, know_state, edit_state, delete_state, report_state, profile_state]:
        state_dict.pop(user_id, None)

    await update.message.reply_text("❌ Диалог отменён. Введите /start для начала работы.", parse_mode=None)

async def handle_profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.callback_query.message if update.callback_query else update.message
    if update.callback_query: await update.callback_query.answer()
    profile_state[update.effective_user.id] = {'step': 0}
    await msg.reply_text(" Просмотр профиля\n\nВведите имя и класс:", parse_mode=None)

async def handle_profile_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = getattr(update.message, 'text', None)
    if not text: await update.message.reply_text("❌ Отправьте текстовое сообщение.", parse_mode=None); return
    state = profile_state[user_id]
    if state['step'] == 0:
        parts = text.split()
        if len(parts) < 2: await update.message.reply_text("❌ Формат: Имя Класс", parse_mode=None); return
        name = parts[0].capitalize()
        try: class_num = int(parts[1])
        except ValueError: await update.message.reply_text("❌ Класс должен быть числом.", parse_mode=None); return
        full_profile = get_student_full_profile(name, class_num)
        if not full_profile: await update.message.reply_text(f"❌ {name} не найден.", parse_mode=None); del profile_state[user_id]; return
        def clean(v):
            if not v: return '—'
            c = v.replace('не указаны, ', '').replace(', не указаны', '').strip()
            return '—' if c.lower() in ('не указаны', '') else c
        msg = f"📂 Профиль: {full_profile['name']} ({full_profile['class']} класс)\n📅 Периоды: {', '.join(full_profile['periods']) or 'Нет'}\n Темы: {clean(full_profile['data'].get('topics'))}\n✅ Успехи: {clean(full_profile['data'].get('successes'))}\n⚠️ Трудности: {clean(full_profile['data'].get('difficulties'))}"
        await update.message.reply_text(msg, parse_mode=None)
        del profile_state[user_id]

# ========== ЗАПУСК ==========
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.post_init = post_init
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("add", handle_add_command))
    app.add_handler(CommandHandler("know", handle_know_command))
    app.add_handler(CommandHandler("edit", handle_edit_command))
    app.add_handler(CommandHandler("delete", handle_delete_command))
    app.add_handler(CommandHandler("list", handle_list_command))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("cancel", handle_cancel))
    app.add_handler(CommandHandler("profile", handle_profile_command))
    app.add_handler(MessageHandler(~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(universal_callback))
    print("\n🚀 БОТ ЗАПУЩЕН!")
    app.run_polling(poll_interval=2)

if __name__ == '__main__':
    main()