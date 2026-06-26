"""
Асинхронный менеджер задач генерации.

Каждый POST /chapter* или /chapters/all немедленно возвращает task_id.
Генерация выполняется в фоне (asyncio.create_task).
Клиент опрашивает GET /task/{task_id}/status и скачивает результат через
GET /task/{task_id}/download по готовности.

Хранилище — in-memory dict; задачи живут TASK_TTL секунд после завершения.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Время хранения завершённой задачи (секунды)
TASK_TTL: int = 3600  # 1 час


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Task:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    message: str = "Задача поставлена в очередь"
    result_bytes: Optional[bytes] = None
    filename: str = "result.zip"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "message": self.message,
            "filename": self.filename if self.status == TaskStatus.DONE else None,
            "error": self.error,
        }


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = asyncio.Lock()

    def _new_id(self) -> str:
        return uuid.uuid4().hex

    async def create(self) -> Task:
        task = Task(task_id=self._new_id())
        async with self._lock:
            self._tasks[task.task_id] = task
        return task

    async def get(self, task_id: str) -> Optional[Task]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def set_running(self, task_id: str, message: str = "Обработка архива...") -> None:
        async with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.status = TaskStatus.RUNNING
                t.message = message

    async def set_done(self, task_id: str, result_bytes: bytes, filename: str) -> None:
        async with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.status = TaskStatus.DONE
                t.message = "Готово — файл доступен для скачивания"
                t.result_bytes = result_bytes
                t.filename = filename
                t.finished_at = time.time()

    async def set_error(self, task_id: str, error: str) -> None:
        async with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.status = TaskStatus.ERROR
                t.message = "Ошибка генерации"
                t.error = error
                t.finished_at = time.time()

    async def cleanup_expired(self) -> None:
        """Удаляет завершённые задачи старше TASK_TTL секунд."""
        now = time.time()
        async with self._lock:
            expired = [
                tid for tid, t in self._tasks.items()
                if t.finished_at and (now - t.finished_at) > TASK_TTL
            ]
            for tid in expired:
                del self._tasks[tid]


# Глобальный синглтон
task_manager = TaskManager()
