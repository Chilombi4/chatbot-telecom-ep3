import os
import json
import tarfile
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 1. Dataset de Telecomunicaciones (Español Chileno - Caso 2)
DATA = [
    # consulta_saldo
    ("quiero cachar cuanto saldo me queda en el celu", "consulta_saldo"),
    ("cuantos gigas me quedan disponibles este mes", "consulta_saldo"),
    ("necesito ver mi bolsa de minutos y datos", "consulta_saldo"),
    ("cuanto me queda de plan antes que se acabe", "consulta_saldo"),
    ("consultar saldo actual por favor", "consulta_saldo"),
    ("me quede sin megas quiero revisar mi consumo", "consulta_saldo"),
    ("dime cuantos datos moviles tengo todavia", "consulta_saldo"),
    ("revisar el consumo de mi linea prepago", "consulta_saldo"),
    ("cuantos minutos libres me quedan en el plan", "consulta_saldo"),
    ("ver saldo y bolsa de internet", "consulta_saldo"),

    # estado_facturacion
    ("cuando vence mi boleta de este mes", "estado_facturacion"),
    ("quiero pagar la cuenta del internet hogar", "estado_facturacion"),
    ("me cobraron de mas en la factura actual", "estado_facturacion"),
    ("necesito descargar mi ultima boleta", "estado_facturacion"),
    ("cuanto tengo que pagar este mes por el plan", "estado_facturacion"),
    ("tengo deuda pendiente en mi servicio", "estado_facturacion"),
    ("porque me subio tanto el monto de la boleta", "estado_facturacion"),
    ("donde veo el detalle de cobros de mi factura", "estado_facturacion"),
    ("quiero saber la fecha de vencimiento de pago", "estado_facturacion"),
    ("ya pague la cuenta pero sigue saliendo morosa", "estado_facturacion"),

    # soporte_tecnico
    ("el internet de la casa anda super lento hoy", "soporte_tecnico"),
    ("se cayo la senal del wifi y no tengo net", "soporte_tecnico"),
    ("no me pescan los datos moviles en el telefono", "soporte_tecnico"),
    ("la luz roja del router esta parpadeando", "soporte_tecnico"),
    ("tengo microcortes todo el rato cuando trabajo", "soporte_tecnico"),
    ("no puedo hacer llamadas dice sin servicio", "soporte_tecnico"),
    ("como reinicio el modem de fibra optica", "soporte_tecnico"),
    ("la velocidad de bajada es pesima ayuda", "soporte_tecnico"),
    ("mi chip no agarra red 5g en santiago", "soporte_tecnico"),
    ("se corto el internet en todo el sector", "soporte_tecnico"),

    # derivacion_humano
    ("quiero hablar con un ejecutivo ahora mismo", "derivacion_humano"),
    ("pasame con una persona real no quiero un bot", "derivacion_humano"),
    ("necesito dar de baja mi plan urgente", "derivacion_humano"),
    ("voy a poner un reclamo en el sernac pesimo servicio", "derivacion_humano"),
    ("quiero portarme a otra compania me aburri", "derivacion_humano"),
    ("comuniqueme con un operador humano por favor", "derivacion_humano"),
    ("necesito renegociar mi contrato comercial", "derivacion_humano"),
    ("quiero cancelar la suscripcion del hogar", "derivacion_humano"),
    ("atencion ejecutiva urgente problema grave", "derivacion_humano"),
    ("no me sirve tu respuesta necesito un agente", "derivacion_humano"),
]

# Multiplicamos y agregamos ligeras variaciones para simular volumen de entrenamiento
df = pd.DataFrame(DATA * 15, columns=["texto", "intencion"])

X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["intencion"], test_size=0.25, random_state=42, stratify=df["intencion"]
)

# 2. Pipeline de NLP + Clasificador Lineal (ideal para interpretabilidad exacta de pesos)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
    ("clf", LogisticRegression(C=2.0, max_iter=500, random_state=42))
])

pipeline.fit(X_train, y_train)

# 3. Evaluación de Métricas (Requeridas en IE10)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

metrics = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1_score": round(float(f1), 4),
    "classes": list(pipeline.classes_)
}

print("=== MÉTRICAS DEL MODELO (IE10) ===")
print(json.dumps(metrics, indent=2))
print("\nReporte detallado:\n", classification_report(y_test, y_pred))

# 4. Guardar artefactos y empaquetar para AWS SageMaker (model.tar.gz)
os.makedirs("artifacts/code", exist_ok=True)
joblib.dump(pipeline, "artifacts/model.joblib")

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Copiar script de inferencia dentro de code/ para que SageMaker lo ejecute automáticamente
import shutil
shutil.copy("inference.py", "artifacts/code/inference.py")

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("artifacts/model.joblib", arcname="model.joblib")
    tar.add("artifacts/metrics.json", arcname="metrics.json")
    tar.add("artifacts/code", arcname="code")

print("\n[OK] Paquete 'model.tar.gz' generado exitosamente listo para S3 y SageMaker.")