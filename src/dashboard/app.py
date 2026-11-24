import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_SECRET_KEY", "telecom-ep3-secret-2026")

st.set_page_config(
    page_title="Chatbot IA Telecomunicaciones - EP3",
    page_icon="",
    layout="wide"
)

st.title(" Sistema Inteligente de Atención al Cliente — Telecomunicaciones")
st.caption("Evaluación Parcial 3 (ITY1102) | Equipo 1: Santiago Vega | Arquitectura Cloud + AWS SageMaker + RabbitMQ")

# --- Barra Lateral: Configuración y Pruebas de Arquitectura ---
with st.sidebar:
    st.header(" Parámetros de Simulación")
    canal = st.selectbox("Canal de Origen (RF01)", ["whatsapp", "web", "app"])
    usuario_id = st.text_input("ID Cliente (CRM Siebel)", value="cli-98765")
    simular_fallo = st.checkbox(
        "Simular caída CRM Siebel (Circuit Breaker)",
        value=False,
        help="Activa el modo degradado para probar tolerancia a fallos (RNF02)"
    )
    st.divider()
    st.markdown("**Estado del Backend:**")
    try:
        health = requests.get(f"{API_URL}/healthz", timeout=2).json()
        st.success(f"API Online ({health['modo_inferencia']})")
    except Exception:
        st.error("API Offline — Verifica que Uvicorn esté corriendo")

# --- Pestañas Principales ---
tab_chat, tab_kpis = st.tabs([" Simulador de Chat e Interpretabilidad (IE2)", " Monitoreo Ejecutivo y Cola RabbitMQ (IE5)"])

with tab_chat:
    col_izq, col_der = st.columns([1.1, 1])

    with col_izq:
        st.subheader("Interacción con el Cliente")
        ejemplos = [
            "Escribir consulta personalizada...",
            "Hola, mi rut es 18.456.789-K y quiero cachar cuantos gigas me quedan",
            "Me cobraron de mas en la boleta de este mes, necesito revisar",
            "Se cayo la senal del wifi y la luz roja del router parpadea",
            "Quiero hablar con un ejecutivo humano ahora mismo por un reclamo"
        ]
        seleccion = st.selectbox("Ejemplos rápidos (Español Chileno):", ejemplos)
        mensaje_input = st.text_area(
            "Mensaje del cliente:",
            value="" if seleccion == ejemplos[0] else seleccion,
            height=100,
            placeholder="Escribe aquí tu consulta..."
        )

        if st.button("Enviar Consulta al Orquestador ", type="primary", use_container_width=True):
            if not mensaje_input.strip():
                st.warning("Por favor ingresa un mensaje.")
            else:
                with st.spinner("Sanitizando PII, invocando SageMaker y publicando en RabbitMQ..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/v1/chat",
                            headers={"X-API-Key": API_KEY},
                            json={
                                "usuario_id": usuario_id,
                                "canal": canal,
                                "mensaje": mensaje_input,
                                "simular_fallo_crm": simular_fallo
                            },
                            timeout=10
                        )
                        if resp.status_code == 200:
                            st.session_state["ultima_respuesta"] = resp.json()
                        else:
                            st.error(f"Error HTTP {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Error de conexión con la API: {e}")

        if "ultima_respuesta" in st.session_state:
            data = st.session_state["ultima_respuesta"]
            st.divider()
            st.markdown("###  Respuesta Generada")
            if data["escalar_humano"]:
                st.warning(f"**[DERIVACIÓN A EJECUTIVO HUMANO - RF05]**\n\n{data['respuesta']}")
            else:
                st.info(data["respuesta"])

            st.markdown("** Seguridad y Privacidad (Ley 19.628):**")
            st.code(f"Prompt Sanitizado: {data['mensaje_sanitizado']}\nPII Detectado: {data['pii_detectado']}")

    with col_der:
        st.subheader(" Interpretabilidad del Modelo (IE2)")
        if "ultima_respuesta" in st.session_state:
            data = st.session_state["ultima_respuesta"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Intención Detectada", data["intencion"])
            m2.metric("Confianza", f"{data['confianza']*100:.1f}%")
            m3.metric("Latencia E2E", f"{data['latencia_ms']} ms")

            st.markdown("**Palabras clave con mayor peso en la decisión:**")
            if data["interpretabilidad"]:
                df_interp = pd.DataFrame(data["interpretabilidad"])
                st.bar_chart(df_interp.set_index("palabra")["peso_impacto"])
            else:
                st.caption("No se detectaron tokens relevantes en el vocabulario base.")

            with st.expander("Ver Trazabilidad Técnica (CRM ACL + RabbitMQ)"):
                st.json({
                    "trace_id": data["trace_id"],
                    "crm_siebel_acl": data["datos_crm_acl"],
                    "evento_rabbitmq_publicado": data["evento_rabbitmq_enviado"]
                })
        else:
            st.info("Envía una consulta a la izquierda para visualizar la explicabilidad del modelo.")

with tab_kpis:
    st.subheader("Indicadores Operativos en Tiempo Real")
    if st.button("Actualizar Métricas "):
        st.rerun()

    try:
        metricas = requests.get(
            f"{API_URL}/v1/metricas",
            headers={"X-API-Key": API_KEY},
            timeout=3
        ).json()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Interacciones", metricas["total_interacciones"])
        k2.metric("Tasa Automatización (Meta ≥60%)", f"{metricas.get('tasa_automatizacion_pct', 0)}%")
        k3.metric("Casos Escalados a Humano", metricas.get("escalados_humano", 0))
        k4.metric("Latencia Promedio (Meta <3000ms)", f"{metricas['latencia_promedio_ms']} ms")

        if metricas["historial"]:
            st.markdown("### Registro Reciente de Eventos Distribuidos")
            df_hist = pd.DataFrame(metricas["historial"])[
                ["trace_id", "canal", "intencion", "confianza", "escalar_humano", "latencia_ms", "evento_rabbitmq_enviado"]
            ]
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.caption("Aún no hay interacciones registradas en esta sesión.")
    except Exception as e:
        st.error(f"No se pudieron cargar las métricas: {e}")