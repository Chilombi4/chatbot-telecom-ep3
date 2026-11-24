import os
import json
import joblib
import numpy as np

# Respuestas base RAG simuladas por intención (Sector Telecomunicaciones)
RESPUESTAS_RAG = {
    "consulta_saldo": "Según tu registro en CRM Siebel, dispones de 18.4 GB de navegación y 420 minutos libres en tu ciclo actual.",
    "estado_facturacion": "Tu última boleta por $24.990 CLP vence el día 15 del mes. Puedes pagarla en el Portal Web o App móvil.",
    "soporte_tecnico": "Detectamos intermitencia en tu nodo. Hemos enviado un pulso de reinicio remoto a tu router; aguarda 2 minutos.",
    "derivacion_humano": "Entiendo tu solicitud. Estoy transfiriendo tu caso con todo el contexto acumulado a un ejecutivo humano."
}

def model_fn(model_dir):
    """Carga el modelo entrenado desde el directorio de SageMaker."""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parsea la petición JSON entrante."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data.get("texto", "")
    raise ValueError(f"Content-Type no soportado: {request_content_type}")

def predict_fn(texto, pipeline):
    """Ejecuta la predicción y extrae la interpretabilidad (contribución de palabras)."""
    probas = pipeline.predict_proba([texto])[0]
    classes = pipeline.classes_
    best_idx = int(np.argmax(probas))
    intencion = str(classes[best_idx])
    confianza = round(float(probas[best_idx]), 4)

    # Interpretabilidad local (contribución de cada palabra a la clase ganadora)
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vector = vectorizer.transform([texto]).toarray()[0]
    class_weights = clf.coef_[best_idx]

    contribuciones = []
    for idx in np.where(tfidf_vector > 0)[0]:
        score = float(tfidf_vector[idx] * class_weights[idx])
        contribuciones.append({
            "palabra": str(feature_names[idx]),
            "peso_impacto": round(score, 4)
        })

    # Ordenar palabras por mayor impacto positivo
    contribuciones = sorted(contribuciones, key=lambda x: x["peso_impacto"], reverse=True)[:5]

    # Regla de negocio EP1 (Sección 1.2.2): Fallback humano si confianza < 0.80 o intención explícita
    escalar_humano = bool(confianza < 0.80 or intencion == "derivacion_humano")
    respuesta = (
        RESPUESTAS_RAG["derivacion_humano"]
        if escalar_humano
        else RESPUESTAS_RAG.get(intencion, "Consulta procesada.")
    )

    return {
        "intencion": intencion,
        "confianza": confianza,
        "escalar_humano": escalar_humano,
        "respuesta": respuesta,
        "interpretabilidad": contribuciones,
        "probabilidades_todas": {str(c): round(float(p), 4) for c, p in zip(classes, probas)}
    }

def output_fn(prediction, response_content_type):
    """Devuelve la respuesta serializada en JSON."""
    if response_content_type == "application/json":
        return json.dumps(prediction, ensure_ascii=False)
    raise ValueError(f"Content-Type de salida no soportado: {response_content_type}")