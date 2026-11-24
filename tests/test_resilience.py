import pytest
from src.api.resilience import CircuitBreaker, CircuitOpenError


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def falla():
    raise ConnectionError("caido")


def ok():
    return "ok"


def test_estado_inicial_cerrado():
    assert CircuitBreaker().state == CircuitBreaker.CLOSED


def test_se_abre_al_superar_umbral():
    cb = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    for _ in range(3):
        with pytest.raises(ConnectionError):
            cb.call(falla)
    assert cb.state == CircuitBreaker.OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(ok)  # rechazada sin ejecutarse


def test_semiabierto_y_recuperacion():
    clk = FakeClock()
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10, clock=clk)
    with pytest.raises(ConnectionError):
        cb.call(falla)
    assert cb.state == CircuitBreaker.OPEN
    clk.t = 11
    assert cb.state == CircuitBreaker.HALF_OPEN
    assert cb.call(ok) == "ok"
    assert cb.state == CircuitBreaker.CLOSED


def test_falla_en_semiabierto_reabre():
    clk = FakeClock()
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=10, clock=clk)
    for _ in range(5):
        with pytest.raises(ConnectionError):
            cb.call(falla)
    clk.t = 11
    with pytest.raises(ConnectionError):
        cb.call(falla)
    assert cb.state == CircuitBreaker.OPEN


def test_exito_reinicia_contador():
    cb = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    with pytest.raises(ConnectionError):
        cb.call(falla)
    cb.call(ok)
    assert cb.failures == 0
