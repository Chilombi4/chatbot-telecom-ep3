# Chatbot inteligente de atención al cliente — Telecomunicaciones

Proyecto académico desarrollado para la asignatura **Arquitectura de Sistemas de Inteligencia Artificial (ITY1102)**. Implementa una API con FastAPI que clasifica consultas de clientes de telecomunicaciones, protege datos personales, consulta una ACL simulada de CRM Siebel y publica eventos en RabbitMQ.

El sistema incluye una interfaz operativa desarrollada con Streamlit y puede ejecutar el modelo con:

- un modelo local serializado con `joblib`, o
- AWS SageMaker, activando inferencia remota mediante variables de entorno.

## Funcionalidades

- Clasificación de cuatro intenciones con TF-IDF y Regresión Logística.
- Interpretabilidad local mediante los términos presentes y su peso de impacto.
- Sanitización de RUT, correo electrónico y tarjetas de crédito antes de la inferencia.
- API Key obligatoria en los endpoints protegidos.
- Limitación de 30 solicitudes por minuto en `POST /v1/chat`.
- Simulación de caída de CRM mediante el patrón Circuit Breaker.
- Publicación opcional de eventos persistentes en RabbitMQ.
- Métricas en memoria para el dashboard.
- Empaquetado del modelo para AWS SageMaker.

## Intenciones reconocidas

| Intención | Uso |
|---|---|
| `consulta_saldo` | Consultas de datos móviles, minutos o saldo |
| `estado_facturacion` | Boletas, cobros, pagos y vencimientos |
| `soporte_tecnico` | Señal, router, fibra, Wi-Fi o velocidad |
| `derivacion_humano` | Solicitudes de atención, reclamos o cancelaciones |

El resultado se deriva a atención humana cuando el modelo predice `derivacion_humano` o cuando la confianza es menor a `0.80`.

## Arquitectura

```text
Streamlit
   |
   | HTTP / JSON
   v
FastAPI
   |
   +--> Sanitización PII
   |
   +--> SageMaker remoto o modelo local joblib
   |
   +--> ACL simulada de CRM Siebel
   |
   +--> RabbitMQ (cola eventos_chatbot_telecom)
```

RabbitMQ y SageMaker son integraciones externas opcionales. Si RabbitMQ no está disponible, la API continúa funcionando y devuelve `evento_rabbitmq_enviado: false`. Si SageMaker no está habilitado, se utiliza el modelo local contenido en `src/model/artifacts/model.joblib`.

## Estructura

```text
chatbot-telecom-ep3/
├── artifacts/
│   ├── model.joblib
│   └── metrics.json
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── worker.py
│   ├── dashboard/
│   │   └── app.py
│   └── model/
│       ├── artifacts/
│       ├── deploy_sagemaker.py
│       ├── inference.py
│       ├── model.tar.gz
│       └── train_and_package.py
├── tests/
│   ├── locustfile.py
│   └── test_unit.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dashboard.txt
└── runtime.txt
```

## Requisitos

- Python 3.11
- pip y un entorno virtual
- Docker Desktop opcional
- RabbitMQ opcional para consumir eventos
- Credenciales y permisos AWS opcionales para SageMaker

## Instalación local

Desde PowerShell:

```powershell
git clone https://github.com/Chilombi4/chatbot-telecom-ep3.git
cd chatbot-telecom-ep3

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
```

## Ejecutar la API

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Documentación disponible en:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Estado: `http://127.0.0.1:8000/healthz`

Todos los endpoints que aparecen en el dashboard requieren:

```text
X-API-Key: telecom-ep3-secret-2026
```

La clave predeterminada puede reemplazarse con la variable `API_SECRET_KEY`.

## Ejecutar el dashboard

En otra terminal, con el entorno virtual activo:

```powershell
streamlit run src/dashboard/app.py
```

Por defecto abre `http://localhost:8501` y se conecta a `http://127.0.0.1:8000`.

Para usar otra API:

```powershell
$env:API_URL="http://127.0.0.1:8000"
$env:API_SECRET_KEY="tu-clave"
streamlit run src/dashboard/app.py
```

## Endpoints principales

### Estado

```http
GET /healthz
```

Indica si el modo activo es `LOCAL_FALLBACK` o `AWS_SAGEMAKER`.

### Procesar mensaje

```http
POST /v1/chat
X-API-Key: telecom-ep3-secret-2026
Content-Type: application/json
```

Ejemplo:

```json
{
  "usuario_id": "cli-98765",
  "canal": "whatsapp",
  "mensaje": "Hola, mi RUT es 18.456.789-K y quiero saber cuántos gigas me quedan",
  "simular_fallo_crm": false
}
```

La respuesta incluye:

- identificador de trazabilidad;
- mensaje sanitizado y tipos de PII encontrados;
- intención y confianza;
- decisión de derivación humana;
- respuesta generada;
- resultado de la consulta simulada al CRM;
- pesos de interpretabilidad;
- latencia;
- estado de publicación en RabbitMQ.

### Métricas

```http
GET /v1/metricas
X-API-Key: telecom-ep3-secret-2026
```

Las métricas se mantienen en memoria y se reinician al reiniciar el proceso FastAPI.

## Seguridad y privacidad

El servicio reemplaza los siguientes datos detectados:

- RUT chileno → `[RUT_ENMASCARADO]`
- correo electrónico → `[EMAIL_ENMASCARADO]`
- tarjeta de crédito → `[TARJETA_ENMASCARADA]`

La protección se realiza mediante expresiones regulares antes de enviar el texto al modelo.

## Ejecutar pruebas

```powershell
pytest -v
```

La suite actual cubre:

- sanitización de RUT y correo;
- autenticación por API Key;
- clasificación de consulta de saldo;
- derivación humana y simulación de caída del CRM.

## Pruebas de carga

Instala Locust como dependencia opcional:

```powershell
pip install locust
locust -f tests/locustfile.py --host http://127.0.0.1:8000
```

Abre `http://localhost:8089` para iniciar la ejecución.

## RabbitMQ

RabbitMQ puede ejecutarse localmente con Docker:

```powershell
docker run --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

Inicia el worker:

```powershell
$env:RABBITMQ_HOST="localhost"
python src/api/worker.py
```

La API se conecta a `RABBITMQ_HOST`, cuyo valor predeterminado es `localhost`.

## Docker

Construir y ejecutar la API:

```powershell
docker build -t chatbot-telecom-api .
docker run --rm -p 8000:8000 -e API_SECRET_KEY="telecom-ep3-secret-2026" chatbot-telecom-api
```

La imagen ejecuta Uvicorn en el puerto `8000`.

## Entrenar y empaquetar el modelo

Desde la raíz del proyecto:

```powershell
cd src/model
python train_and_package.py
cd ../..
```

El script genera o actualiza:

- `artifacts/model.joblib`;
- `artifacts/metrics.json`;
- `model.tar.gz`;
- el código de inferencia listo para SageMaker.

Las métricas generadas en el artefacto actual son `accuracy`, `precision`, `recall` y `f1_score`, todas con valor `1.0`.

## Desplegar en AWS SageMaker

El repositorio incluye `src/model/deploy_sagemaker.py`, que crea o intenta crear el modelo, la configuración y el endpoint en AWS SageMaker.

Antes de ejecutar el script se requieren credenciales AWS y una imagen disponible en la región configurada. La cuenta, el bucket y el rol usados deben existir o tener permisos suficientes.

Para activar SageMaker desde la API:

```powershell
$env:USE_SAGEMAKER="true"
$env:AWS_REGION="us-east-1"
$env:SAGEMAKER_ENDPOINT="chatbot-telecom-endpoint-v1"
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Sin `USE_SAGEMAKER=true`, la API continúa usando el modelo local.

## Variables de entorno

| Variable | Valor predeterminado | Uso |
|---|---|---|
| `API_SECRET_KEY` | `telecom-ep3-secret-2026` | API Key |
| `USE_SAGEMAKER` | `false` | Activa inferencia SageMaker |
| `SAGEMAKER_ENDPOINT` | `chatbot-telecom-endpoint-v1` | Endpoint SageMaker |
| `AWS_REGION` | `us-east-1` | Región AWS |
| `RABBITMQ_HOST` | `localhost` | Host RabbitMQ |
| `API_URL` | `http://127.0.0.1:8000` | URL usada por Streamlit |

## Autor

- Santiago Vega
- Equipo 1
- Evaluación Parcial 3
