import os
import re
import time
import json
import joblib
import boto3
import pika
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configuración mediante variables de entorno
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "telecom-ep3-secret-2026")
USE_SAGEMAKER = os.getenv("USE_SAGEMAKER", "false").lower() == "true"
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "chatbot-telecom-endpoint-v1")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

# Inicializar Rate Limiter (IE7: Seguridad)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="API Orquestador - Chatbot Inteligente Telecomunicaciones (EP3)",
    description="Microservicio Orquestador con Sanitización PII (Ley 19.628), Inferencia AWS SageMaker, ACL Siebel y mensajería RabbitMQ.",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Cargar modelo local como respaldo cuando USE_SAGEMAKER=false
local_pipeline = None
if not USE_SAGEMAKER:
    model_path = os.path.join(os.path.dirname(__file__), "../model/artifacts/model.joblib")
    if os.path.exists(model_path):
        local_pipeline = joblib.load(model_path)

sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION) if USE_SAGEMAKER else None

# Historial en memoria para visualización rápida en Dashboard
HISTORIAL_INTERACCIONES = []

# --- Esquemas Pydantic (Validación de Contratos) ---
class ChatRequest(BaseModel):
    usuario_id: str = Field(..., example="cli-98765")
    canal: str = Field(..., example="whatsapp", description="Canales soportados: whatsapp, web, app")
    mensaje: str = Field(..., example="Hola, mi rut es 18.456.789-K y quiero cachar cuanto saldo me queda")
    simular_fallo_crm: bool = Field(default=False, description="Activa el patrón Circuit Breaker de Siebel para testing")

class ChatResponse(BaseModel):
    trace_id: str
    canal: str
    mensaje_original_recibido: bool
    mensaje_sanitizado: str
    pii_detectado: list[str]
    intencion: str
    confianza: float
    escalar_humano: bool
    respuesta: str
    datos_crm_acl: dict
    interpretabilidad: list[dict]
    latencia_ms: float
    evento_rabbitmq_enviado: bool

# --- Seguridad: Autenticación por API Key (IE7) ---
def verificar_api_key(x_api_key: str = Header(..., description="Clave de acceso empresarial")):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida o no autorizada")
    return x_api_key

# --- Componente 1: Microservicio Sanitizador PII (Ley 19.628) ---
def sanitizar_pii(texto: str) -> tuple[str, list[str]]:
    detectados = []
    texto_limpio = texto

    # Detectar RUT chileno (ej: 12.345.678-9 o 12345678-K)
    patron_rut = r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"
    if re.search(patron_rut, texto_limpio):
        texto_limpio = re.sub(patron_rut, "[RUT_ENMASCARADO]", texto_limpio)
        detectados.append("RUT_CHILENO")

    # Detectar Email
    patron_email = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    if re.search(patron_email, texto_limpio):
        texto_limpio = re.sub(patron_email, "[EMAIL_ENMASCARADO]", texto_limpio)
        detectados.append("EMAIL")

    # Detectar Tarjeta de Crédito (16 dígitos)
    patron_tarjeta = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    if re.search(patron_tarjeta, texto_limpio):
        texto_limpio = re.sub(patron_tarjeta, "[TARJETA_ENMASCARADA]", texto_limpio)
        detectados.append("TARJETA_CREDITO")

    return texto_limpio, detectados

# --- Componente 2: Conector ACL CRM Siebel + Circuit Breaker ---
def consultar_crm_siebel_acl(usuario_id: str, intencion: str, simular_fallo: bool = False) -> dict:
    if simular_fallo:
        # Patrón Circuit Breaker abierto: entrega respuesta degradada sin colapsar el sistema
        return {
            "estado_crm": "CIRCUIT_BREAKER_OPEN",
            "mensaje": "CRM Siebel no disponible temporalmente. Operando en modo contingencia."
        }
    return {
        "estado_crm": "CONNECTED_ACL",
        "cliente_id": usuario_id,
        "plan_activo": "Plan Fibra + Móvil 5G Full",
        "estado_cuenta": "Al día" if intencion != "estado_facturacion" else "Boleta emitida ($24.990)"
    }

# --- Componente 3: Publicador de Eventos en RabbitMQ (Event-Driven) ---
def publicar_evento_rabbitmq(evento: dict) -> bool:
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, connection_attempts=1, socket_timeout=1))
        channel = connection.channel()
        channel.queue_declare(queue="eventos_chatbot_telecom", durable=True)
        channel.basic_publish(
            exchange="",
            routing_key="eventos_chatbot_telecom",
            body=json.dumps(evento, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2)  # Mensaje persistente
        )
        connection.close()
        return True
    except Exception:
        return False

# --- Componente 4: Motor de Inferencia (AWS SageMaker o Fallback Local) ---
def ejecutar_inferencia(texto_sanitizado: str) -> dict:
    if USE_SAGEMAKER and sagemaker_runtime:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"texto": texto_sanitizado})
        )
        return json.loads(response["Body"].read().decode("utf-8"))
    else:
        # Importar lógica idéntica a inference.py para pruebas locales sin costo AWS
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
        from inference import predict_fn
        return predict_fn(texto_sanitizado, local_pipeline)

# --- Endpoints REST ---
@app.get("/healthz", tags=["Observabilidad"])
def health_check():
    return {
        "status": "healthy",
        "modo_inferencia": "AWS_SAGEMAKER" if USE_SAGEMAKER else "LOCAL_FALLBACK",
        "endpoint": SAGEMAKER_ENDPOINT if USE_SAGEMAKER else "local_joblib"
    }

@app.post("/v1/chat", response_model=ChatResponse, tags=["Orquestador IA"])
@limiter.limit("30/minute")
def procesar_mensaje(
    request: Request,
    payload: ChatRequest,
    api_key: str = Depends(verificar_api_key)
):
    inicio = time.perf_counter()
    trace_id = f"trc-{int(time.time() * 1000)}"

    if payload.canal.lower() not in ["whatsapp", "web", "app"]:
        raise HTTPException(status_code=400, detail="Canal no soportado. Use: whatsapp, web o app")

    # 1. Sanitización PII
    texto_limpio, pii_lista = sanitizar_pii(payload.mensaje)

    # 2. Inferencia en AWS SageMaker (con interpretabilidad)
    prediccion = ejecutar_inferencia(texto_limpio)

    # 3. Consulta CRM Siebel vía ACL
    datos_crm = consultar_crm_siebel_acl(
        payload.usuario_id,
        prediccion["intencion"],
        payload.simular_fallo_crm
    )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)

    # 4. Publicar evento asíncrono a RabbitMQ
    tipo_evento = "CasoEscaladoHumano" if prediccion["escalar_humano"] else "ConversacionFinalizada"
    evento_mq = {
        "trace_id": trace_id,
        "timestamp": datetime.utcnow().isoformat(),
        "tipo_evento": tipo_evento,
        "canal": payload.canal,
        "usuario_id": payload.usuario_id,
        "intencion": prediccion["intencion"],
        "confianza": prediccion["confianza"],
        "escalado": prediccion["escalar_humano"],
        "latencia_ms": latencia_ms
    }
    mq_ok = publicar_evento_rabbitmq(evento_mq)

    resultado = ChatResponse(
        trace_id=trace_id,
        canal=payload.canal,
        mensaje_original_recibido=True,
        mensaje_sanitizado=texto_limpio,
        pii_detectado=pii_lista,
        intencion=prediccion["intencion"],
        confianza=prediccion["confianza"],
        escalar_humano=prediccion["escalar_humano"],
        respuesta=prediccion["respuesta"],
        datos_crm_acl=datos_crm,
        interpretabilidad=prediccion["interpretabilidad"],
        latencia_ms=latencia_ms,
        evento_rabbitmq_enviado=mq_ok
    )

    HISTORIAL_INTERACCIONES.append(resultado.model_dump())
    return resultado

@app.get("/v1/metricas", tags=["Observabilidad"])
def obtener_metricas_negocio(api_key: str = Depends(verificar_api_key)):
    total = len(HISTORIAL_INTERACCIONES)
    if total == 0:
        return {"total_interacciones": 0, "tasa_automatizacion": 0, "latencia_promedio_ms": 0, "historial": []}
    
    escalados = sum(1 for x in HISTORIAL_INTERACCIONES if x["escalar_humano"])
    automatizados = total - escalados
    latencia_prom = round(sum(x["latencia_ms"] for x in HISTORIAL_INTERACCIONES) / total, 2)

    return {
        "total_interacciones": total,
        "automatizados": automatizados,
        "escalados_humano": escalados,
        "tasa_automatizacion_pct": round((automatizados / total) * 100, 1),
        "latencia_promedio_ms": latencia_prom,
        "historial": HISTORIAL_INTERACCIONES[-20:]
    }