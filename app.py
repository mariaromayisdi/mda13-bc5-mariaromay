# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un analista de datos experto en Python.

Tu tarea es generar código Python para responder preguntas sobre un DataFrame llamado df.

IMPORTANTE:

- Debes devolver un JSON válido con esta estructura:
    {
        "tipo": "grafico",
        "codigo": "...",
        "interpretacion": "..."
    }

- Si la pregunta no se puede responder:
    {
        "tipo": "fuera_de_alcance",
        "codigo": "",
        "interpretacion": "..."
    }

REGLAS:

- El campo "codigo" debe contener SOLO código Python ejecutable
- NO incluyas explicaciones dentro del código
- NO uses print()
- NO inventes columnas
- El código debe crear una variable llamada fig

LIBRERÍAS:

- Usa pandas (pd)
- Usa plotly.express (px) o plotly.graph_objects (go)
- NO uses matplotlib

DATASET:

El DataFrame df contiene:

- ts (datetime)
- ms_played (int)
- track_name (str)
- artist_name (str)
- album_name (str)
- platform (str)
- shuffle (bool)
- skipped (bool)
- hour (int)
- weekday (int)
- month (int)
- year (int)
- duration_min (float)

INSTRUCCIONES:

- Rankings → groupby + sum/count + sort_values(ascending=False)
- Limitar con head(10)
- Evolución → agrupar por tiempo
- Patrones → usar hour, weekday
- Comparaciones → filtrar por periodos

VISUALIZACIÓN:

- Usa px.bar() para rankings
- Usa px.line() para evolución
- La figura debe guardarse en una variable llamada fig

EJEMPLO:

fig = px.bar(df_grouped, x="artist_name", y="duration_min")

ERRORES:

Si no se puede responder:
devuelve tipo "fuera_de_alcance"
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------
    
    df = df[df['master_metadata_track_name'].notna()]

    # Convertir timestamp
    df['ts'] = pd.to_datetime(df['ts'])

    # Renombrar columnas (más simple para el LLM)
    df['track_name'] = df['master_metadata_track_name']
    df['artist_name'] = df['master_metadata_album_artist_name']
    df['album_name'] = df['master_metadata_album_album_name']

    # Duración en minutos
    df['duration_min'] = df['ms_played'] / 60000

    # Variables temporales
    df['hour'] = df['ts'].dt.hour
    df['weekday'] = df['ts'].dt.weekday
    df['month'] = df['ts'].dt.month
    df['year'] = df['ts'].dt.year

    # Fin de semana (muy útil para preguntas tipo C)
    df['is_weekend'] = df['weekday'] >= 5

    # Limpiar skipped (null → False)
    df['skipped'] = df['skipped'].fillna(False)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT
    
    #.format(
        #fecha_min=fecha_min,
        # fecha_max=fecha_max,
        # plataformas=plataformas,
         #reason_start_values=reason_start_values,
         #reason_end_values=reason_end_values,
    # )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?

        # La aplicación sigue una arquitectura text-to-code en la que el
        # LLM no recibe los datos directamente, sino únicamente la
        # estructura del DataFrame y la pregunta del usuario.
        #
        # En cada interacción, el modelo recibe el system prompt (con las
        # columnas disponibles y las reglas de generación) y la pregunta
        # del usuario. A partir de esto, genera código Python en formato
        # texto, encapsulado dentro de un JSON con los campos "tipo",
        # "codigo" e "interpretacion".
        #
        # El código generado no se ejecuta en el modelo, sino en local,
        # dentro de la aplicación, mediante exec(), utilizando el DataFrame
        # df previamente cargado. Este código debe crear una variable "fig"
        # con una visualización de Plotly que luego se renderiza.
        #
        # El LLM no recibe los datos directamente para evitar exponer
        # información sensible, reducir el consumo de tokens y garantizar
        # que el procesamiento real se realiza en un entorno controlado.

# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.

        # El system prompt es fundamental porque define completamente cómo
        # debe comportarse el modelo. En mi caso, le proporciono la
        # estructura exacta del DataFrame, las librerías permitidas
        # (pandas y plotly), el formato de salida obligatorio (JSON) y la
        # instrucción de generar una variable "fig".
        #
        # Además, incluyo reglas explícitas como no inventar columnas,
        # usar groupby para agregaciones y limitar resultados con head(10),
        # lo que guía al modelo hacia respuestas correctas.
        #
        # Por ejemplo, la pregunta “Top 5 artistas más escuchados” funciona
        # correctamente porque el prompt indica agrupar por artista,
        # ordenar y generar un gráfico con Plotly en una variable "fig".
        #
        # Si eliminara la instrucción de usar Plotly o de crear "fig", el
        # modelo podría devolver código con matplotlib o sin variable de
        # salida, lo que rompería la ejecución. Esto demuestra que un
        # prompt incompleto o ambiguo provoca errores en toda la cadena.


# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
        # El usuario introduce una pregunta en la interfaz de Streamlit.
        # Esa pregunta se envía junto con el system prompt al modelo de
        # OpenAI.
        #
        # El modelo genera una respuesta en formato JSON que contiene el
        # tipo de respuesta, el código Python a ejecutar y una breve
        # interpretación.
        #
        # La aplicación limpia y parsea ese JSON, extrae el código y lo
        # ejecuta en local sobre el DataFrame df.
        #
        # El código genera una figura de Plotly almacenada en la variable
        # "fig". Esta figura se devuelve a la interfaz y se renderiza como
        # visualización interactiva.
        #
        # Finalmente, también se muestra la interpretación textual junto
        # al código generado, completando la respuesta al usuario.