from fastapi.testclient import TestClient
from src.api.main import app, sanitizar_pii

client = TestClient(app)
HEADERS = {"X-API-Key": "telecom-ep3-secret-2026"}

def test_sanitizacion_pii_rut_y_email():
    texto = "Mi rut es 18.456.789-K y mi correo es santiago@duocuc.cl"
    limpio, detectados = sanitizar_pii(texto)
    assert "[RUT_ENMASCARADO]" in limpio
    assert "[EMAIL_ENMASCARADO]" in limpio
    assert "RUT_CHILENO" in detectados
    assert "EMAIL" in detectados

def test_autenticacion_invalida():
    resp = client.post("/v1/chat", headers={"X-API-Key": "clave-falsa"}, json={
        "usuario_id": "u1",
        "canal": "web",
        "mensaje": "hola"
    })
    assert resp.status_code == 401

def test_flujo_completo_consulta_saldo():
    resp = client.post("/v1/chat", headers=HEADERS, json={
        "usuario_id": "cli-100",
        "canal": "whatsapp",
        "mensaje": "quiero cachar cuantos gigas me quedan"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intencion"] == "consulta_saldo"
    assert len(data["interpretabilidad"]) > 0

def test_derivacion_humano_y_circuit_breaker():
    resp = client.post("/v1/chat", headers=HEADERS, json={
        "usuario_id": "cli-200",
        "canal": "app",
        "mensaje": "quiero hablar con un ejecutivo humano urgente",
        "simular_fallo_crm": True
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalar_humano"] is True
    assert data["datos_crm_acl"]["estado_crm"] == "CIRCUIT_BREAKER_OPEN"