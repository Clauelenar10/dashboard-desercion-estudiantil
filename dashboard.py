
import streamlit as st
from pymongo import MongoClient
st.set_page_config(page_title="Dashboard Deserción")

# Título
st.title(" Dashboard de Deserción Estudiantil")
st.markdown("---")

CONNECTION_STRING = st.secrets["CONNECTION_STRING"]

# Nombres
DATABASE_NAME = "Estudiantes"
COLLECTION_NAME = "Estudiantes_Materias"
JSON_FILE = "estudiantes_documentos.json"
# Conectar
@st.cache_resource
def get_connection():
    client = MongoClient(CONNECTION_STRING)
    return client

client = get_connection()
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
# Extraer todos los datos
@st.cache_data
def load_data():
    data = list(collection.find({}))
    return data

datos = load_data()

# Convertir a DataFrame
import pandas as pd

# Aplanar los datos para análisis
registros = []
for doc in datos:
    registro = {
        '_id': doc['_id'],
        'edad': doc['datos_personales']['edad'],
        'genero': doc['datos_personales']['genero'],
        'estrato': doc['datos_personales']['estrato'],
        'programa': doc['academico']['programa'],
        'semestre_actual': doc['academico']['semestre_actual'],
        'promedio': doc['metricas_rendimiento']['promedio_acumulado'],
        'materias_cursadas': doc['metricas_rendimiento']['materias_cursadas_total'],
        'materias_perdidas': doc['metricas_rendimiento']['materias_perdidas_total'],
        'materias_repetidas': doc['metricas_rendimiento']['materias_repetidas'],
        'becado': doc['estado']['becado'],
        'desertor': doc['estado']['desertor'],
        'ciudad': doc['location']['ciudad'],
        'es_barranquilla': doc['location']['es_barranquilla'],
        'periodo': doc['periodo_info']['ultimo_periodo'],
        'icfes_matematicas': doc['ICFES'].get('matematicas'),
        'icfes_lectura': doc['ICFES'].get('lectura_critica'),
        'icfes_ciencias': doc['ICFES'].get('ciencias'),
    }
    registros.append(registro)

df = pd.DataFrame(registros)
total_estudiantes = collection.count_documents({})

# KPIs principales
st.subheader("📊 Métricas Generales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Total Estudiantes", len(df))

with col2:
    tasa_desercion = (df['desertor'].sum() / len(df)) * 100
    st.metric("⚠️ Tasa Deserción", f"{tasa_desercion:.1f}%")

with col3:
    promedio_general = df['promedio'].mean()
    st.metric("📝 Promedio General", f"{promedio_general:.2f}")

with col4:
    total_becados = df['becado'].value_counts().get('Institucional', 0) + df['becado'].value_counts().get('Oficial', 0)
    st.metric("🎓 Becados", total_becados)

st.markdown("---")
# Distribución de deserción
st.subheader("📉 Distribución de Deserción")

col1, col2 = st.columns(2)

with col1:
    # Pie chart deserción
    import plotly.express as px
    
    desercion_counts = df['desertor'].value_counts()
    desercion_counts.index = ['No Desertor', 'Desertor']
    
    fig = px.pie(values=desercion_counts.values, 
                 names=desercion_counts.index,
                 title="Deserción vs No Deserción",
                 color_discrete_sequence=['#00cc96', '#ef553b'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Bar chart por periodo
    desercion_periodo = df.groupby(['periodo', 'desertor']).size().reset_index(name='count')
    desercion_periodo['desertor'] = desercion_periodo['desertor'].map({0: 'No Desertor', 1: 'Desertor'})
    
    fig2 = px.bar(desercion_periodo, x='periodo', y='count', color='desertor',
                  title="Deserción por Periodo",
                  barmode='group',
                  color_discrete_map={'No Desertor': '#00cc96', 'Desertor': '#ef553b'})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")