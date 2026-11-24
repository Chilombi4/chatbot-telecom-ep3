from fastapi.testclient import TestClient
from src.api.main import app, sanitizar_pii, crm_breaker

client = TestClient(app)
import os
HEADERS = {"X-API-Key": os.getenv("API_SECRET_KEY", "telecom-ep3-secret-2026")}

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
    crm_breaker.reset()
    resp = client.post("/v1/chat", headers=HEADERS, json={
        "usuario_id": "cli-200",
        "canal": "app",
        "mensaje": "quiero hablar con un ejecutivo humano urgente",
        "simular_fallo_crm": True
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalar_humano"] is True
    assert data["datos_crm_acl"]["estado_crm"] == "CRM_ERROR_DEGRADADO"  # 1er fallo: modo contingencia, circuito aun cerrado


def test_circuit_breaker_se_abre_tras_fallos_consecutivos():
    crm_breaker.reset()
    estados = []
    for i in range(crm_breaker.failure_threshold + 1):
        r = client.post("/v1/chat", headers=HEADERS, json={
            "usuario_id": f"cli-{i}", "canal": "web",
            "mensaje": "se cayo el internet", "simular_fallo_crm": True})
        assert r.status_code == 200  # el sistema nunca cae
        estados.append(r.json()["datos_crm_acl"]["estado_crm"])
    assert estados[0] == "CRM_ERROR_DEGRADADO"
    assert estados[-1] == "CIRCUIT_BREAKER_OPEN"
    crm_breaker.reset()


def test_validacion_canal_no_soportado():
    r = client.post("/v1/chat", headers=HEADERS, json={"usuario_id": "u", "canal": "fax", "mensaje": "hola"})
    assert r.status_code == 400


def test_healthz_y_metricas_requieren_formato_correcto():
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/metricas").status_code == 422  # falta cabecera X-API-Key
    assert client.get("/v1/metricas", headers=HEADERS).status_code == 200


def test_mensaje_invalido_retorna_422():
    r = client.post("/v1/chat", headers=HEADERS, json={"usuario_id": "u"})
    assert r.status_code == 422