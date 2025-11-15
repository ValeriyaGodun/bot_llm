import asyncio
import logging
import os
import re

from huggingface_hub import InferenceClient
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from conversation_utils import BotConfig, ConversationManager

# Настраиваем минимальный уровень логирования, чтобы видеть, что происходит на работающем боте
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Читаем конфигурацию из окружения, чтобы легко менять параметры без правок кода
def load_config() -> BotConfig:
    telegram_token = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN").strip()
    model_name = os.environ.get("HF_MODEL_NAME", "HuggingFaceTB/SmolLM3-3B").strip()
    hf_token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN").strip()
    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "Ты - полезный ассистент. Отвечай на русском языке.",
    )
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "400"))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    top_p = float(os.environ.get("TOP_P", "0.9"))
    repetition_penalty = float(os.environ.get("REPETITION_PENALTY", "1.1"))
    history_max_pairs = int(os.environ.get("HISTORY_MAX_PAIRS", "4"))
    history_max_tokens = int(os.environ.get("HISTORY_MAX_TOKENS", "2048"))
    history_ttl_seconds = int(os.environ.get("HISTORY_TTL_SECONDS", "3600"))
    model_device = os.environ.get("MODEL_DEVICE", "auto").lower()

    return BotConfig(
        telegram_token=telegram_token,
        hf_token=hf_token,
        model_name=model_name,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        history_max_pairs=history_max_pairs,
        history_max_tokens=history_max_tokens,
        history_ttl_seconds=history_ttl_seconds,
        model_device=model_device,
    )


CONFIG = load_config()

if CONFIG.hf_token == "YOUR_HF_TOKEN" or not CONFIG.hf_token:
    logger.warning("HF_TOKEN не задан. Будет выполнена попытка скачать публичную модель без токена.")

if CONFIG.telegram_token == "YOUR_BOT_TOKEN" or not CONFIG.telegram_token:
    raise RuntimeError("TELEGRAM_TOKEN не задан. Установите переменную окружения TELEGRAM_TOKEN.")

BOT_TOKEN = CONFIG.telegram_token

# Используем один клиент Hugging Face на весь процесс, чтобы не открывать соединения лишний раз
CLIENT = InferenceClient(token=CONFIG.hf_token or None)

logger.info("Использую модель %s через Hugging Face Inference API.", CONFIG.model_name)

# ConversationManager хранит историю диалогов и следит за лимитами токенов
conversation_manager = ConversationManager(tokenizer=None, config=CONFIG)

# Очищаем ответ от тегов <think>, которые модель добавляет в ответах
def _clean_model_output(text: str) -> str:

    if "<think>" not in text.lower():
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()

# Генерируем ответ для конкретного пользователя
def generate_response(user_id: int, user_message: str) -> str:

    # Сохраняем пользовательское сообщение в историю до обращения к модели
    conversation_manager.add_user_message(user_id, user_message)
    messages = conversation_manager.get_history(user_id)

    try:
        # Выполняем запрос к HF Inference API
        completion = CLIENT.chat.completions.create(
            model=CONFIG.model_name,
            messages=messages,
            max_tokens=CONFIG.max_new_tokens,
            temperature=CONFIG.temperature,
            top_p=CONFIG.top_p,
        )
        content = ""
        if completion.choices:
            message_obj = completion.choices[0].message  
            if isinstance(message_obj, dict):
                content = message_obj.get("content", "") or ""
            else:
                content = getattr(message_obj, "content", "") or ""
        response = _clean_model_output(content.strip())
    except Exception as exc:
        logger.exception("Ошибка при запросе к Hugging Face Inference API: %s", exc)
        return "Извините, не удалось обработать запрос."

    if response:
        # Ответ тоже кладем в историю, чтобы поддерживать контекст диалога
        conversation_manager.add_assistant_message(user_id, response)
        return response

    return "Извините, не удалось обработать запрос."
    
# Обрабатываем команду /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    welcome_text = """
🤖 Привет! Я бот на основе AI-модели SmolLM3-3B от Hugging Face.

Просто напиши мне сообщение, и я постараюсь помочь!

Команды:
/start - показать это сообщение
/clear - очистить историю диалога
/help - помощь
    """.strip()
    await update.message.reply_text(welcome_text)

# Обрабатываем команду /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    help_text = (
        "ℹ️ Помощь по боту\n\n"
        "Я использую языковую модель для генерации ответов. Вот что нужно знать:\n\n"
        "- Я помню историю наших последних сообщений\n"
        "- Если ответы стали странными, используйте /clear\n"
        "- Ответы генерируются автоматически и могут содержать ошибки\n"
        "- Бот работает лучше с конкретными вопросами"
    )
    await update.message.reply_text(help_text)

 # Обрабатываем команду /clear
async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_manager.clear_history(user_id)
    if update.message is None:
        return
    logger.info("История пользователя %s очищена по команде.", user_id)
    await update.message.reply_text("История диалога очищена!")

# Обрабатываем текстовое сообщение
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    user_id = update.effective_user.id
    text = (update.message.text or "").strip()

    if not text:
        await update.message.reply_text("Пожалуйста, отправьте текстовое сообщение.")
        return

    # Показываем индикатор набора, чтобы пользователь видел, что бот обрабатывает запрос
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Запускаем генерацию ответа в отдельном потоке
    loop = asyncio.get_running_loop()
    try:
        logger.info("Получено сообщение от %s: %s", user_id, text[:60])
        response = await loop.run_in_executor(None, generate_response, user_id, text)
        await update.message.reply_text(response)
        logger.info("Ответ пользователю %s отправлен успешно.", user_id)
    except Exception as exc:
        logger.exception("Ошибка при обработке сообщения от %s: %s", user_id, exc)
        await update.message.reply_text("Произошла ошибка при обработке запроса. Попробуйте позже.")

# Обрабатываем нетекстовое сообщение
async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text("Пожалуйста, отправьте текстовое сообщение.")


def main() -> None:
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("TELEGRAM_TOKEN не задан. Установите переменную окружения TELEGRAM_TOKEN.")

    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(~filters.TEXT & (~filters.COMMAND), handle_non_text))

    logger.info("Бот запущен и готов принимать сообщения.")
    application.run_polling()


if __name__ == "__main__":
    main()