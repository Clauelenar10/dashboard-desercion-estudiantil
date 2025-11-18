
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
st.subheader(" Métricas Generales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Total Estudiantes", len(df))

with col2:
    tasa_desercion = (df['desertor'].sum() / len(df)) * 100
    st.metric(" Tasa Deserción", f"{tasa_desercion:.1f}%")

with col3:
    promedio_general = df['promedio'].mean()
    st.metric(" Promedio General", f"{promedio_general:.2f}")

with col4:
    total_becados = df['becado'].value_counts().get('Institucional', 0) + df['becado'].value_counts().get('Oficial', 0)
    st.metric("🎓 Becados", total_becados)

st.markdown("---")
# Distribución de deserción
st.subheader("Distribución de Deserción")

# Filtro por estrato
estratos_disponibles = sorted(df['estrato'].dropna().unique())
estratos_seleccionados = st.multiselect(
    "Filtrar por Estrato:",
    [f"Estrato {int(e)}" for e in estratos_disponibles],
    default=[f"Estrato {int(e)}" for e in estratos_disponibles]
)

# Filtrar datos según selección
if len(estratos_seleccionados) == 0:
    st.warning("Por favor selecciona al menos un estrato")
    st.stop()
elif len(estratos_seleccionados) == len(estratos_disponibles):
    df_filtrado = df.copy()
    estrato_label = "Todos los Estratos"
else:
    estratos_nums = [int(e.split()[-1]) for e in estratos_seleccionados]
    df_filtrado = df[df['estrato'].isin(estratos_nums)].copy()
    estrato_label = ", ".join(estratos_seleccionados)

# Filtrar solo los 3 periodos válidos
df_filtrado = df_filtrado[df_filtrado['periodo'].isin([202410, 202430, 202510])]

# Validar que hay datos después del filtro
if len(df_filtrado) == 0:
    st.warning("No hay datos disponibles para esta selección")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart deserción
        import plotly.express as px
        
        desercion_counts = df_filtrado['desertor'].value_counts()
        
        if len(desercion_counts) > 0:
            # Crear diccionario para mapear
            labels = []
            values = []
            for idx, val in desercion_counts.items():
                labels.append('No Desertor' if idx == 0 else 'Desertor')
                values.append(val)
            
            fig = px.pie(values=values, 
                         names=labels,
                         title=f"Deserción - {estrato_label}",
                         color_discrete_sequence=['#00cc96', '#ef553b'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Total estudiantes: {len(df_filtrado)}")
        else:
            st.warning("No hay datos para mostrar") 
    
    with col2:
    # Bar chart por periodo
        desercion_periodo = df_filtrado.groupby(['periodo', 'desertor']).size().reset_index(name='count')
        
        if len(desercion_periodo) > 0:
            desercion_periodo['desertor'] = desercion_periodo['desertor'].map({0: 'No Desertor', 1: 'Desertor'})
            
            fig2 = px.bar(desercion_periodo, x='periodo', y='count', color='desertor',
                        title=f"Deserción por Periodo - {estrato_label}",
                        barmode='group',
                        color_discrete_map={'No Desertor': '#00cc96', 'Desertor': '#ef553b'})
            
            # Forzar eje X como categórico
            fig2.update_xaxes(type='category', categoryorder='array', categoryarray=['202410', '202430', '202510'])
            
            # Quitar controles deslizantes
            fig2.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No hay datos para este periodo")

st.markdown("---")