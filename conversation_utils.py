from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from transformers import PreTrainedTokenizerBase 
except ImportError:
    PreTrainedTokenizerBase = object  


# Словарь соответствует формату сообщений chat-completion API (role + content)
ChatMessage = Dict[str, str]


@dataclass
class BotConfig:
    # Единая точка хранения всех параметров бота, чтобы удобно передавать их в другие классы
    telegram_token: str = "" # ключ Telegram-бота
    hf_token: str = ""  # ключ Hugging Face Inference
    model_name: str = "HuggingFaceTB/SmolLM3-3B"  # идентификатор модели в HF
    system_prompt: str = "Ты - полезный ассистент. Отвечай на русском языке. Используй не более 400 токенов."  # базовый промпт для модели
    max_new_tokens: int = 400  # ограничение на длину генерируемого ответа
    temperature: float = 0.7  # параметр стохастичности, чем выше значение, тем выше вероятность случайности и ответы будут более разнообразными, чем ниже значение, тем более детерминированным и более предсказуемым будет ответ
    top_p: float = 0.9  # nucleus sampling, коэффициент для ограничения вероятности выбора токенов
    repetition_penalty: float = 1.1  # штраф за повторения
    history_max_pairs: int = 4  # максимум пар user/assistant, которые сохраняем
    history_max_tokens: int = 2048  # лимит токенов на историю
    history_ttl_seconds: int = 3600  # время жизни истории без активности
    model_device: str = "auto"  # выбор устройства при локальном запуске


@dataclass
class ConversationEntry:
    # Структура для хранения истории конкретного пользователя 
    history: List[ChatMessage] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class ConversationManager:
    # Управляет историей диалогов и следит за тем, чтобы история не выходила за лимиты
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase], config: BotConfig):
        self._tokenizer = tokenizer
        self._config = config
        self._store: Dict[int, ConversationEntry] = {}

    def _now(self) -> float:
        return time.time()

    # Удаляет истории, которые не обновлялись дольше заданного времени жизни (TTL), чтобы не раздувать память
    def purge_inactive(self) -> None:
        ttl = self._config.history_ttl_seconds
        if ttl <= 0:
            return

        current = self._now()
        inactive = [
            user_id
            for user_id, entry in self._store.items()
            if current - entry.updated_at > ttl
        ]
        for user_id in inactive:
            del self._store[user_id]
            
    # Инициализация хранилища для нового пользователя
    def _ensure_entry(self, user_id: int) -> ConversationEntry:        
        entry = self._store.get(user_id)
        if entry is None:
            entry = ConversationEntry(
                history=[{"role": "system", "content": self._config.system_prompt}],
                updated_at=self._now(),
            )
            self._store[user_id] = entry
        return entry

    # Добавляет реплику пользователя в историю и сокращает историю, если она превысила лимиты
    def add_user_message(self, user_id: int, content: str) -> None:        
        self.purge_inactive()
        entry = self._ensure_entry(user_id)
        entry.history.append({"role": "user", "content": content})
        entry.updated_at = self._now()
        self._truncate_history(entry.history)

    # Сохраняет ответ модели в ту же историю, чтобы поддерживать контекст
    def add_assistant_message(self, user_id: int, content: str) -> None:
        entry = self._ensure_entry(user_id)
        entry.history.append({"role": "assistant", "content": content})
        entry.updated_at = self._now()
        self._truncate_history(entry.history)
        
    # Формируем промпт в формате токенайзера либо в простом текстовом виде
    def build_prompt(self, user_id: int, add_generation_prompt: bool = True) -> str:
        entry = self._ensure_entry(user_id)
        history = entry.history

        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        prompt_parts: List[str] = []
        for msg in history:
            if msg["role"] == "system":
                prompt_parts.append(f"<|system|>\n{msg['content']}\n")
            elif msg["role"] == "user":
                prompt_parts.append(f"<|user|>\n{msg['content']}\n")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg['content']}\n")

        if add_generation_prompt:
            prompt_parts.append("<|assistant|>\n")

        return "".join(prompt_parts)

    # Возвращает копию текущей истории, чтобы вызывающий код не мог испортить оригинал
    def get_history(self, user_id: int) -> List[ChatMessage]:
        return list(self._ensure_entry(user_id).history)

    # Полностью удаляет историю пользователя, например по команде /clear
    def clear_history(self, user_id: int) -> None:
        if user_id in self._store:
            del self._store[user_id]

    # Ограничивает количество пар user/assistant в истории
    def _truncate_history(self, history: List[ChatMessage]) -> None:
        max_pairs = self._config.history_max_pairs
        if max_pairs > 0:

            max_messages = 1 + max_pairs * 2
            if len(history) > max_messages:
                # Сохраняем системное сообщение и последние пары user/assistant
                system = history[0]
                history[:] = [system] + history[-(max_messages - 1):]

        # Ограничиваем количество токенов в истории
        max_tokens = self._config.history_max_tokens
        if max_tokens <= 0:
            return

        while len(history) > 1:
            prompt = self.build_prompt_from_history(history)
            token_count = self._count_tokens(prompt)
            if token_count <= max_tokens:
                break
            # Удаляем самую старую пару user/assistant (если есть)
            if len(history) > 2:
                del history[1:3]
            else:
                break

    # Собирает промпт из уже подготовленной истории
    def build_prompt_from_history(
        self,
        history: List[ChatMessage],
        add_generation_prompt: bool = True,
    ) -> str:
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        prompt_parts: List[str] = []
        for msg in history:
            if msg["role"] == "system":
                prompt_parts.append(f"<|system|>\n{msg['content']}\n")
            elif msg["role"] == "user":
                prompt_parts.append(f"<|user|>\n{msg['content']}\n")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg['content']}\n")

        if add_generation_prompt:
            prompt_parts.append("<|assistant|>\n")

        return "".join(prompt_parts)

    # Подсчитываем токены через реальный токенайзер, либо делим строку по пробелам, если токенайзер не передан
    def _count_tokens(self, prompt: str) -> int:
        if not prompt:
            return 0

        if self._tokenizer:
            encoded = self._tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            input_ids = encoded.get("input_ids", [])
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            return len(input_ids)

        return len(prompt.split())

