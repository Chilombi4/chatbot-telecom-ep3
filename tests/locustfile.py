import random
from locust import HttpUser, task, between

CONSULTAS = [
    "quiero cachar cuanto saldo me queda en el celu",
    "cuando vence mi boleta de este mes",
    "se cayo la senal del wifi y no tengo net",
    "mi rut es 15.234.567-8 quiero revisar mi factura"
]

class ChatbotUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def enviar_consulta(self):
        self.client.post(
            "/v1/chat",
            headers={"X-API-Key": "telecom-ep3-secret-2026"},
            json={
                "usuario_id": f"cli-{random.randint(1000, 9999)}",
                "canal": random.choice(["whatsapp", "web", "app"]),
                "mensaje": random.choice(CONSULTAS)
            }
        )