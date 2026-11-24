import time
import boto3
from sagemaker import image_uris  # Si no instalas sagemaker, usa la URI directa abajo

region = "us-east-1"
boto_session = boto3.Session(region_name=region)
s3 = boto_session.client("s3")
sm = boto_session.client("sagemaker")
sts = boto_session.client("sts")

account_id = sts.get_caller_identity()["Account"]
role_arn = f"arn:aws:iam::{account_id}:role/LabRole"
bucket_name = f"sagemaker-{region}-{account_id}"
s3_key = "chatbot-telecom-ep3/model.tar.gz"

# 1. Crear bucket S3 si no existe y subir model.tar.gz
try:
    s3.create_bucket(Bucket=bucket_name)
except Exception:
    pass  # El bucket ya existe

s3.upload_file("model.tar.gz", bucket_name, s3_key)
model_data_url = f"s3://{bucket_name}/{s3_key}"
print(f"[OK] Modelo subido a: {model_data_url}")

# Imagen oficial de Scikit-Learn 1.2-1 en us-east-1
sklearn_image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

model_name = "chatbot-telecom-model-v1"
endpoint_config_name = "chatbot-telecom-config-v1"
endpoint_name = "chatbot-telecom-endpoint-v1"

# 2. Crear Modelo en SageMaker
sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": sklearn_image,
        "ModelDataUrl": model_data_url,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": model_data_url
        }
    },
    ExecutionRoleArn=role_arn
)
print(f"[OK] Modelo creado: {model_name}")

# 3. Crear Configuración del Endpoint
sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.t2.medium"
    }]
)

# 4. Desplegar Endpoint
sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
print(f"Desplegando endpoint '{endpoint_name}'... esperando estado InService...")

waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=endpoint_name)
print(f"\n[ÉXITO] Endpoint operativo en AWS SageMaker: {endpoint_name}")