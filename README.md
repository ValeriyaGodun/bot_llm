# LLM Telegram Bot

Телеграм-бот, который общается с пользователем, опираясь на языковую модель Hugging Face: https://huggingface.co/HuggingFaceTB/SmolLM3-3B  

Проект состоит из двух ключевых модулей:
- `llm_bot.py` - вся логика Telegram: обработчики команд, вызов модели.
- `conversation_utils.py` - управление историей диалога, конфигурацией и ограничениями.

## Запуск 

1. Установите зависимости:
   ```bash
   pip install python-telegram-bot==21.4 huggingface_hub
   ```
2. Определите переменные окружения:
   ```bash
   export TELEGRAM_TOKEN="123:ABC"
   export HF_TOKEN="hf_..."
   export HF_MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
   ```
3. Запустите бота:
   ```bash
   python llm_bot.py
   ```

## Конфигурация

Когда запускается функция `load_config()`, она берёт значения параметров из переменных окружения, а затем передаёт их в `BotConfig`. Таким образом, все параметры конфигурации бота (например, токены, имя модели, ограничения истории) собираются в одном объекте, что упрощает настройку и использование этих данных по всему коду.

Особенности:

* `telegram_token`, `hf_token` - обязательные токены, без них запуск запрещается;
* `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty` - настройка генерации;
* `history_max_pairs`, `history_max_tokens`, `history_ttl_seconds` - ограничения истории и её времени жизни (TTL).

## Основная архитектура

1. Пользователь отправляет сообщение и `python-telegram-bot` вызывает `handle_message`.
2. `handle_message` проверяет текст, показывает индикатор набора (`ChatAction.TYPING`) и передаёт задачу дальше `loop.run_in_executor(None, generate_response, ...)`. 
3. Функция `generate_response` запускается в отдельном потоке, чтобы не блокировать основной, пока идёт запрос к Hugging Face.`generate_response`:
   - записывает сообщение пользователя в `ConversationManager`;
   - собирает историю `ConversationManager.get_history`;
   - обращается к `InferenceClient.chat.completions.create`;
   - очищает ответ от `<think>` и сохраняет  реплику ассистента в историю.
4. Telegram-бот отправляет ответ пользователю.

## Conversation Utils

Файл `conversation_utils.py`содержит основную логику работы с историей:

- `purge_inactive()` выбрасывает диалоги больше заданного TTL, чтобы не расходовать память.
- `_ensure_entry()` создаёт запись для пользователя, добавляя системный промпт в начало.
- `_truncate_history()` применяет два ограничения:
  * по количеству пар user/assistant (`history_max_pairs`);
  * по количеству токенов (`history_max_tokens`). Для подсчёта токенов используется реальный токенайзер, если он передан, иначе `len(prompt.split())`.
    
Таким образом, при любой длине переписки модель получает не превышающий лимиты контекст.

## Hugging Face Inference

Клиент создаётся один раз:

```python
CLIENT = InferenceClient(token=CONFIG.hf_token or None)
```

Далее используется метод `chat.completions.create` с параметрами из `BotConfig`. При желании можно заменить модель, передав другое имя.

Чтобы скрыть «глубокое мышление» модели, введена функция `_clean_model_output`, которая срезает блоки `<think>...</think>`.

## Управление командами

- `/start` - приветственный текст и краткое описание.
- `/help` - правила работы с ботом, упоминание истории и автоматичности ответов.
- `/clear` - ручная очистка истории пользователя через `conversation_manager.clear_history`.
- Хендлеры разделены на текстовые и нетекстовые, чтобы пользователь получал понятные ошибки.
