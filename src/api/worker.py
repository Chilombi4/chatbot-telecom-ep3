import os
import json
import time
import pika

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

def callback(ch, method, properties, body):
    evento = json.loads(body.decode("utf-8"))
    print(f"\n[RabbitMQ Worker] Evento recibido: {evento['tipo_evento']}")
    print(f"  -> Trace ID: {evento['trace_id']} | Canal: {evento['canal']} | Intención: {evento['intencion']} ({evento['confianza']*100:.1f}%)")
    if evento["escalado"]:
        print("  -> [ALERTA EJECUTIVO] Derivando conversación con contexto acumulado a agente humano...")
    else:
        print("  -> [MLOps CT] Interacción almacenada para reentrenamiento continuo.")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def iniciar_worker():
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
            channel = connection.channel()
            channel.queue_declare(queue="eventos_chatbot_telecom", durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue="eventos_chatbot_telecom", on_message_callback=callback)
            print(f"[*] Worker RabbitMQ escuchando en '{RABBITMQ_HOST}'. Esperando eventos...")
            channel.start_consuming()
        except Exception as e:
            print(f"[!] Esperando conexión con RabbitMQ ({e})... reintentando en 5s")
            time.sleep(5)

if __name__ == "__main__":
    iniciar_worker()