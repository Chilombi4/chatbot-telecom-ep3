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

    # --- Frases adicionales (v2): 10 por clase para ampliar la diversidad ---
    # consulta_saldo
    ("cuanto me queda de datos moviles", "consulta_saldo"),
    ("puedes decirme mi saldo disponible", "consulta_saldo"),
    ("cuantos minutos de llamadas me quedan", "consulta_saldo"),
    ("quiero saber cuanto he consumido de mi plan", "consulta_saldo"),
    ("me quedan gigas para navegar", "consulta_saldo"),
    ("ver el detalle de mi consumo de datos", "consulta_saldo"),
    ("cuanto saldo tengo en mi chip", "consulta_saldo"),
    ("cuando se renueva mi bolsa de internet y cuanto queda", "consulta_saldo"),
    ("revisar cuantos megas he gastado", "consulta_saldo"),
    ("consulta de saldo de mi numero", "consulta_saldo"),
    # estado_facturacion
    ("cuanto debo de la cuenta del mes", "estado_facturacion"),
    ("quiero ver mi boleta pendiente", "estado_facturacion"),
    ("no entiendo un cobro en mi factura", "estado_facturacion"),
    ("puedo pagar mi cuenta en linea", "estado_facturacion"),
    ("cuando me llega la boleta", "estado_facturacion"),
    ("tengo un cargo duplicado en mi cuenta", "estado_facturacion"),
    ("necesito el comprobante de mi ultimo pago", "estado_facturacion"),
    ("hasta cuando puedo pagar sin recargo", "estado_facturacion"),
    ("me llego una boleta mas cara de lo normal", "estado_facturacion"),
    ("quiero saber el total a pagar de mi factura", "estado_facturacion"),
    # soporte_tecnico
    ("mi internet esta muy lento desde ayer", "soporte_tecnico"),
    ("el router no prende y no tengo conexion", "soporte_tecnico"),
    ("se corta la llamada cada rato", "soporte_tecnico"),
    ("no tengo senal en mi celular", "soporte_tecnico"),
    ("el wifi se desconecta solo", "soporte_tecnico"),
    ("la fibra esta con falla y no navega", "soporte_tecnico"),
    ("mi telefono no se conecta a la red movil", "soporte_tecnico"),
    ("hay intermitencia en el servicio de internet", "soporte_tecnico"),
    ("no carga ninguna pagina y el modem tiene luz roja", "soporte_tecnico"),
    ("la senal 4g esta pesima en mi casa", "soporte_tecnico"),
    # derivacion_humano
    ("quiero conversar con un agente de atencion", "derivacion_humano"),
    ("necesito que me llame un ejecutivo de ventas", "derivacion_humano"),
    ("esto es un reclamo formal quiero un supervisor", "derivacion_humano"),
    ("quiero terminar mi contrato con la compania", "derivacion_humano"),
    ("prefiero hablar con una persona y no con un bot", "derivacion_humano"),
    ("me pueden transferir con un asesor humano", "derivacion_humano"),
    ("voy a reclamar en la subtel por el mal servicio", "derivacion_humano"),
    ("necesito cambiar el titular del contrato con un ejecutivo", "derivacion_humano"),
    ("quiero cancelar mi servicio hoy mismo", "derivacion_humano"),
    ("transfiereme con alguien de carne y hueso", "derivacion_humano"),
]

# Variaciones de superficie (saludos / cierres neutros) para simular mensajes reales.
# Se aplican DESPUES de separar train/test, por frase base, para evitar fuga de datos.
PREFIJOS = ["", "hola, ", "buenas tardes, ", "oye ", "disculpa, "]
SUFIJOS = ["", " por favor", " gracias"]

def expandir(frases):
    filas = []
    for texto, intencion in frases:
        for pre in PREFIJOS:
            for suf in SUFIJOS:
                filas.append((f"{pre}{texto}{suf}".strip(), intencion, texto))
    return filas

# 2. Division por FRASE BASE (GroupShuffleSplit): ninguna frase de prueba, ni sus variaciones,
#    aparece en el entrenamiento. Esto evita la fuga de datos de la version anterior.
base = pd.DataFrame(DATA, columns=["texto", "intencion"])
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import confusion_matrix

filas = pd.DataFrame(expandir(DATA), columns=["texto", "intencion", "base"])
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
idx_train, idx_test = next(gss.split(filas["texto"], filas["intencion"], groups=filas["base"]))
X_train, y_train = filas.loc[idx_train, "texto"], filas.loc[idx_train, "intencion"]
X_test, y_test = filas.loc[idx_test, "texto"], filas.loc[idx_test, "intencion"]
assert not set(filas.loc[idx_train, "base"]) & set(filas.loc[idx_test, "base"]), "Fuga de datos"

# 3. Pipeline de NLP + Clasificador Lineal (ideal para interpretabilidad exacta de pesos)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
    ("clf", LogisticRegression(C=2.0, max_iter=500, random_state=42))
])

pipeline.fit(X_train, y_train)

# 4. Evaluación de Métricas (Requeridas en IE10)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

metrics = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1_score": round(float(f1), 4),
    "classes": list(pipeline.classes_),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "frases_base_unicas": int(len(DATA)),
    "split": "GroupShuffleSplit por frase base (sin fuga train/test)",
    "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(pipeline.classes_)).tolist(),
}
cv = cross_val_score(pipeline, filas["texto"], filas["intencion"], groups=filas["base"],
                     cv=GroupKFold(n_splits=5), scoring="f1_weighted")
metrics["cv5_f1_weighted_mean"] = round(float(cv.mean()), 4)
metrics["cv5_f1_weighted_std"] = round(float(cv.std()), 4)

print("=== MÉTRICAS DEL MODELO (IE10) ===")
print(json.dumps(metrics, indent=2))
print("\nReporte detallado:\n", classification_report(y_test, y_pred))

# 5. Guardar artefactos y empaquetar para AWS SageMaker (model.tar.gz)
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