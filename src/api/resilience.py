"""Patrón Circuit Breaker (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) para dependencias externas.

Implementación sin dependencias externas para poder probarla de forma aislada.
"""
import threading
import time
from typing import Callable, Any


class CircuitOpenError(Exception):
    """Se lanza cuando el circuito está abierto y la llamada se rechaza sin ejecutarse."""


class CircuitBreaker:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 15.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_state()
            return self._state

    @property
    def failures(self) -> int:
        return self._failures

    def _refresh_state(self) -> None:
        if self._state == self.OPEN and self._clock() - self._opened_at >= self.recovery_timeout:
            self._state = self.HALF_OPEN

    def reset(self) -> None:
        with self._lock:
            self._state, self._failures, self._opened_at = self.CLOSED, 0, 0.0

    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Ejecuta func. Si el circuito está abierto lanza CircuitOpenError sin llamarla."""
        with self._lock:
            self._refresh_state()
            if self._state == self.OPEN:
                raise CircuitOpenError("Circuito abierto")
        try:
            result = func(*args, **kwargs)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                    self._state = self.OPEN
                    self._opened_at = self._clock()
            raise
        with self._lock:
            self._state, self._failures = self.CLOSED, 0
        return result
