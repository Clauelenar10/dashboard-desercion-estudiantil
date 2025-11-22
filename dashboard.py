import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os

st.set_page_config(
    page_title="Dashboard Deserción Estudiantil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================================
st.sidebar.markdown("""
<div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0; font-size: 1.8em;">📊</h1>
    <h2 style="color: white; margin: 5px 0 0 0; font-size: 1.2em;">Sistema de Análisis</h2>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("Dashboard Deserción")
st.sidebar.markdown("### Periodo: 2025-10")
st.sidebar.markdown("---")

# Menú de navegación
seccion = st.sidebar.radio(
    "Seleccione una sección:",
    [
        "1. Características Generales",
        "2. Desertores vs No Desertores",
        "3. Modelo Predictivo"
    ],
    index=0
)

st.sidebar.markdown("---")

CONNECTION_STRING = st.secrets["CONNECTION_STRING"]

# Nombres
DATABASE_NAME = "Estudiantes"
COLLECTION_NAME = "Estudiantes_Materias"

# Conectar
@st.cache_resource
def get_connection():
    client = MongoClient(CONNECTION_STRING)
    return client

client = get_connection()
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Cargar modelo de Keras y metadatos
@st.cache_resource
def load_keras_model():
    """Carga el modelo de Keras guardado y sus metadatos"""
    try:
        model_path = "mejor_modelo_desercion.keras"
        info_path = "mejor_modelo_info.json"
        
        if not os.path.exists(model_path):
            st.warning(f"⚠️ No se encontró el modelo en {model_path}")
            return None, None
        
        # Cargar modelo
        model = keras.models.load_model(model_path)
        
        # Cargar info del modelo
        info = None
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
        
        return model, info
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el modelo: {str(e)}")
        return None, None

modelo_keras, info_modelo = load_keras_model()

# Extraer todos los datos
@st.cache_data
def load_data():
    data = list(collection.find({}))
    return data

datos = load_data()

# Aplanar los datos para análisis completo
registros = []
for doc in datos:
    registro = {
        '_id': doc['_id'],
        # Datos personales
        'edad': doc['datos_personales'].get('edad'),
        'genero': doc['datos_personales'].get('genero', ''),
        'estrato': doc['datos_personales'].get('estrato'),
        'discapacidad': doc['datos_personales'].get('discapacidad', ''),
        # Académico
        'programa': doc['academico'].get('programa', ''),
        'programa_secundario': doc['academico'].get('programa_secundario'),
        'semestre_actual': doc['academico'].get('semestre_actual'),
        'tipo_estudiante': doc['academico'].get('tipo_estudiante', ''),
        'tipo_admision': doc['academico'].get('tipo_admision', ''),
        'estado_academico': doc['academico'].get('estado_academico', ''),
        # Ubicación
        'ciudad': doc['location'].get('ciudad', ''),
        'departamento': doc['location'].get('departamento', ''),
        'pais': doc['location'].get('pais', ''),
        'es_barranquilla': doc['location'].get('es_barranquilla', 0),
        'es_colombia': doc['location'].get('es_colombia', 0),
        # Colegio
        'tipo_colegio': doc['colegio'].get('tipo_colegio'),
        'calendario_colegio': doc['colegio'].get('calendario_colegio'),
        'descripcion_bachillerato': doc['colegio'].get('descripcion_bachillerato'),
        # ICFES
        'puntaje_total': doc['ICFES'].get('puntaje_total'),
        'icfes_matematicas': doc['ICFES'].get('matematicas'),
        'icfes_lectura': doc['ICFES'].get('lectura_critica'),
        'icfes_sociales': doc['ICFES'].get('sociales'),
        'icfes_ciencias': doc['ICFES'].get('ciencias'),
        'icfes_ingles': doc['ICFES'].get('ingles'),
        # Métricas rendimiento
        'promedio': doc['metricas_rendimiento'].get('promedio_acumulado'),
        'materias_cursadas': doc['metricas_rendimiento'].get('materias_cursadas_total', 0),
        'materias_perdidas': doc['metricas_rendimiento'].get('materias_perdidas_total', 0),
        'materias_repetidas': doc['metricas_rendimiento'].get('materias_repetidas', 0),
        # Estado
        'becado': doc['estado'].get('becado', ''),
        'graduado': doc['estado'].get('graduado', 0),
        'desertor': doc['estado'].get('desertor', 0),
        # Periodo
        'periodo': doc['periodo_info'].get('ultimo_periodo'),
    }
    registros.append(registro)

df = pd.DataFrame(registros)

# Mapeo de nombres de departamentos (usado en múltiples secciones)
mapeo_departamentos = {
    'ATLANTICO': 'ATLÁNTICO',
    'BOLIVAR': 'BOLÍVAR',
    'BOGOTA': 'BOGOTÁ D.C.',
    'BOGOTA D.C.': 'BOGOTÁ D.C.',
    'BOGOTÁ': 'BOGOTÁ D.C.',
    'CORDOBA': 'CÓRDOBA',
    'NARINO': 'NARIÑO',
    'QUINDIO': 'QUINDÍO',
    'VALLE': 'VALLE DEL CAUCA',
    'NORTE SANTANDER': 'NORTE DE SANTANDER',
    'ARCHIPIELAGO DE SAN ANDRES': 'ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA',
    'SAN ANDRES': 'ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA'
}

# ============================================================================
# SECCIÓN 1: CARACTERÍSTICAS GENERALES DE LA POBLACIÓN
# ============================================================================
if "1. Características Generales" in seccion:
    st.title("Características Generales de la Población")
    st.markdown("### Análisis descriptivo de toda la población estudiantil")
    st.markdown("---")

    # Métricas principales de población con estilo de fondo
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_estudiantes = len(df)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Total Estudiantes</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{total_estudiantes:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_graduados = df['graduado'].sum()
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Graduados</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{int(total_graduados):,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        estrato_promedio = df['estrato'].mean()
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Estrato Promedio</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{estrato_promedio:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        edad_promedio = df['edad'].mean()
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Edad Promedio</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{edad_promedio:.1f} años</h2>
        </div>
        """, unsafe_allow_html=True)

    # Becados con estilo de fondo
    st.subheader("Becados")
    col1, col2, col3 = st.columns(3)
    
    # Calcular totales de becados
    becados_institucional = (df['becado'] == 'Institucional').sum()
    becados_oficial = (df['becado'] == 'oficial').sum()
    total_becados = becados_institucional + becados_oficial

    with col1:
        pct_total_becados = (total_becados / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Total Becados</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{total_becados:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_total_becados:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pct_becados_inst = (becados_institucional / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Becados Institucional</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{becados_institucional:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_becados_inst:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pct_becados_ofi = (becados_oficial / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Becados Oficial</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{becados_oficial:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_becados_ofi:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Ubicación geográfica con estilo de fondo
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        estudiantes_barranquilla = (df['es_barranquilla'] == 1).sum()
        pct_barranquilla = (estudiantes_barranquilla / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Barranquilla</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{estudiantes_barranquilla:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_barranquilla:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        estudiantes_no_barranquilla = (df['es_barranquilla'] == 0).sum()
        pct_no_barranquilla = (estudiantes_no_barranquilla / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Otras Ciudades</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{estudiantes_no_barranquilla:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_no_barranquilla:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        estudiantes_colombia = (df['es_colombia'] == 1).sum()
        pct_colombia = (estudiantes_colombia / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Colombia</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{estudiantes_colombia:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_colombia:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        estudiantes_extranjero = (df['es_colombia'] == 0).sum()
        pct_extranjero = (estudiantes_extranjero / total_estudiantes * 100)
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #262730; margin: 0;">Extranjero</h3>
            <h2 style="color: #4a4a4a; margin: 10px 0 0 0;">{estudiantes_extranjero:,}</h2>
            <p style="color: #666; margin: 5px 0 0 0;">{pct_extranjero:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================================
    # SECCIÓN 2: DISTRIBUCIÓN GEOGRÁFICA
    # ============================================================================
    st.header("Distribución Geográfica")

    # Mapa de Colombia por departamento (sin Atlántico)
    st.subheader("Estudiantes por Departamento")

    # Filtrar solo estudiantes de Colombia
    df_colombia = df[df['es_colombia'] == 1].copy()

    # Contar por departamento
    estudiantes_depto = df_colombia.groupby('departamento').agg({
        '_id': 'count'
    }).reset_index()
    estudiantes_depto.columns = ['departamento', 'total_estudiantes']
    estudiantes_depto['porcentaje'] = (estudiantes_depto['total_estudiantes'] / estudiantes_depto['total_estudiantes'].sum() * 100).round(2)

    # Normalizar nombres de departamentos
    estudiantes_depto['departamento'] = estudiantes_depto['departamento'].str.upper().str.strip()
    estudiantes_depto['departamento'] = estudiantes_depto['departamento'].replace(mapeo_departamentos)

    # Separar Atlántico para el mapa
    estudiantes_mapa = estudiantes_depto[estudiantes_depto['departamento'] != 'ATLÁNTICO'].copy()

    # Cargar GeoJSON
    @st.cache_data
    def load_geojson():
        url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
        response = requests.get(url)
        return response.json()

    geojson_colombia = load_geojson()

    # Crear el mapa con degradado de color y porcentaje
    fig_mapa = px.choropleth_mapbox(
        estudiantes_mapa,
        geojson=geojson_colombia,
        locations='departamento',
        featureidkey="properties.NOMBRE_DPT",
        color='total_estudiantes',
        color_continuous_scale="Viridis",
        hover_name='departamento',
        hover_data={
            'departamento': False,
            'total_estudiantes': ':,',
            'porcentaje': ':.2f'
        },
        mapbox_style="carto-positron",
        zoom=4.5,
        center={"lat": 4.5, "lon": -74},
        opacity=0.8,
        labels={
            'total_estudiantes': 'Estudiantes', 
            'porcentaje': '% del Total'
        }
    )

    fig_mapa.update_layout(
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_mapa, use_container_width=True)

    st.info("Nota: Atlántico fue excluido del mapa para mejor visualización de otros departamentos.")

    st.markdown("---")

    # Distribución por ciudad del Atlántico
    st.subheader("Estudiantes del Atlántico por Ciudad")

    df_atlantico = df[df['departamento'].str.upper().str.strip().str.contains('ATLANTICO|ATLÁNTICO', na=False)].copy()

    if len(df_atlantico) > 0:
        # Contar por ciudad
        estudiantes_ciudad = df_atlantico.groupby('ciudad').agg({
            '_id': 'count'
        }).reset_index()
        estudiantes_ciudad.columns = ['ciudad', 'total_estudiantes']
        estudiantes_ciudad['porcentaje'] = (estudiantes_ciudad['total_estudiantes'] / estudiantes_ciudad['total_estudiantes'].sum() * 100).round(2)
        
        # Normalizar nombres de ciudades
        estudiantes_ciudad['ciudad'] = estudiantes_ciudad['ciudad'].str.title().str.strip()
        
        # Separar Barranquilla del resto
        barranquilla_data = estudiantes_ciudad[estudiantes_ciudad['ciudad'] == 'Barranquilla']
        otras_ciudades = estudiantes_ciudad[estudiantes_ciudad['ciudad'] != 'Barranquilla'].copy()
        
        # Mostrar Barranquilla como métrica destacada
        if len(barranquilla_data) > 0:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                bq_total = int(barranquilla_data['total_estudiantes'].iloc[0])
                bq_pct = barranquilla_data['porcentaje'].iloc[0]
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 30px; border-radius: 15px; text-align: center;">
                    <h2 style="color: #262730; margin: 0;">Barranquilla</h2>
                    <h1 style="color: #2e7d32; margin: 10px 0;">{bq_total:,}</h1>
                    <h3 style="color: #666; margin: 0;">{bq_pct:.1f}% del Atlántico</h3>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Mostrar gráfico solo para otras ciudades
        if len(otras_ciudades) > 0:
            st.subheader("Otras Ciudades del Atlántico")
            
            # Ordenar por frecuencia
            otras_ciudades = otras_ciudades.sort_values('total_estudiantes', ascending=True)
            
            # Crear gráfico de barras horizontales
            fig_ciudades = px.bar(
                otras_ciudades,
                y='ciudad',
                x='total_estudiantes',
                text='porcentaje',
                orientation='h',
                labels={'ciudad': 'Ciudad', 'total_estudiantes': 'Frecuencia'},
                color='total_estudiantes',
                color_continuous_scale='Blues'
            )
            
            fig_ciudades.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig_ciudades.update_layout(
                height=max(400, len(otras_ciudades) * 25),
                showlegend=False,
                xaxis_title="Número de Estudiantes",
                yaxis_title="",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_ciudades, use_container_width=True)
    else:
        st.warning("No hay datos de estudiantes en Atlántico")

# ============================================================================
# SECCIÓN 2: DESERTORES VS NO DESERTORES
# ============================================================================
elif "2. Desertores vs No Desertores" in seccion:
    st.title("Análisis Comparativo: Desertores vs No Desertores")
    st.markdown("### Comparación detallada entre estudiantes desertores y no desertores")
    st.markdown("---")

    # Filtrar estudiantes sin graduados
    df_sin_graduados = df[df['graduado'] == 0].copy()

    # Tasa de deserción general (grande) con cuadro gris y letras rojas
    tasa_desercion_general = (df_sin_graduados['desertor'].sum() / len(df_sin_graduados) * 100)

    st.markdown("### Tasa de Deserción General")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 30px; border-radius: 15px; text-align: center;">
            <h2 style="color: #d32f2f; margin: 0;">Tasa de Deserción</h2>
            <h1 style="color: #d32f2f; margin: 10px 0; font-size: 3em;">{tasa_desercion_general:.2f}%</h1>
            <h3 style="color: #666; margin: 0;">{df_sin_graduados['desertor'].sum():,} de {len(df_sin_graduados):,} estudiantes</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Deserción por becados con cuadros de fondo
    st.subheader("Deserción por Tipo de Beca")

    col1, col2, col3 = st.columns(3)

    with col1:
        # No becados
        df_no_becados = df_sin_graduados[df_sin_graduados['becado'] == 'No becado']
        if len(df_no_becados) > 0:
            tasa_no_becados = (df_no_becados['desertor'].sum() / len(df_no_becados) * 100)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">No Becados</h3>
                <h2 style="color: #0068c9; margin: 10px 0 0 0;">{tasa_no_becados:.2f}%</h2>
                <p style="color: #666; margin: 5px 0 0 0;">{len(df_no_becados):,} estudiantes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">No Becados</h3>
                <h2 style="color: #0068c9; margin: 10px 0 0 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Becados institucional
        df_bec_inst = df_sin_graduados[df_sin_graduados['becado'] == 'Institucional']
        if len(df_bec_inst) > 0:
            tasa_bec_inst = (df_bec_inst['desertor'].sum() / len(df_bec_inst) * 100)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">Becados Institucional</h3>
                <h2 style="color: #0068c9; margin: 10px 0 0 0;">{tasa_bec_inst:.2f}%</h2>
                <p style="color: #666; margin: 5px 0 0 0;">{len(df_bec_inst):,} estudiantes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">Becados Institucional</h3>
                <h2 style="color: #0068c9; margin: 10px 0 0 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        # Becados oficial
        df_bec_ofi = df_sin_graduados[df_sin_graduados['becado'] == 'oficial']
        if len(df_bec_ofi) > 0:
            tasa_bec_ofi = (df_bec_ofi['desertor'].sum() / len(df_bec_ofi) * 100)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">Becados Oficial</h3>
                <h2 style="color: #0068c9; margin: 10px 0 0 0;">{tasa_bec_ofi:.2f}%</h2>
                <p style="color: #666; margin: 5px 0 0 0;">{len(df_bec_ofi):,} estudiantes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">Becados Oficial</h3>
                <h2 style="color: #0068c9; margin: 10px 0 0 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Deserción por programas
    st.subheader("Deserción por Programa")

    desercion_programa = df_sin_graduados.groupby('programa').agg({
        '_id': 'count',
        'desertor': 'sum'
    }).reset_index()
    desercion_programa.columns = ['programa', 'total', 'desertores']
    desercion_programa['tasa_desercion'] = (desercion_programa['desertores'] / desercion_programa['total'] * 100).round(2)
    desercion_programa = desercion_programa.sort_values('tasa_desercion', ascending=True)

    # Gráfico de barras horizontales
    fig_programas = go.Figure()

    fig_programas.add_trace(go.Bar(
        y=desercion_programa['programa'],
        x=desercion_programa['total'],
        name='Total',
        orientation='h',
        marker_color='lightblue',
        text=desercion_programa['total'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Total: %{x}<extra></extra>'
    ))

    fig_programas.add_trace(go.Bar(
        y=desercion_programa['programa'],
        x=desercion_programa['desertores'],
        name='Desertores',
        orientation='h',
        marker_color='salmon',
        text=desercion_programa['tasa_desercion'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Desertores: %{x}<br>Tasa: %{text}<extra></extra>'
    ))

    fig_programas.update_layout(
        barmode='overlay',
        height=max(600, len(desercion_programa) * 20),
        xaxis_title="Número de Estudiantes",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='y unified'
    )

    st.plotly_chart(fig_programas, use_container_width=True)

    st.markdown("---")

    # Comparación de estratos
    st.subheader("Comparación de Estratos: Desertores vs No Desertores")

    df_desertores = df_sin_graduados[df_sin_graduados['desertor'] == 1]
    df_no_desertores = df_sin_graduados[df_sin_graduados['desertor'] == 0]

    estratos_desertores = df_desertores.groupby('estrato').size().reset_index(name='count')
    estratos_desertores['tipo'] = 'Desertores'

    estratos_no_desertores = df_no_desertores.groupby('estrato').size().reset_index(name='count')
    estratos_no_desertores['tipo'] = 'No Desertores'

    estratos_comparacion = pd.concat([estratos_desertores, estratos_no_desertores])

    fig_estratos = px.bar(
        estratos_comparacion,
        x='estrato',
        y='count',
        color='tipo',
        barmode='group',
        labels={'estrato': 'Estrato', 'count': 'Número de Estudiantes'},
        color_discrete_map={'Desertores': '#ef553b', 'No Desertores': '#00cc96'}
    )

    fig_estratos.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_estratos, use_container_width=True)

    st.markdown("---")
    
    # Tasa de deserción por departamento
    st.subheader("Tasa de Deserción por Departamento")

    df_colombia_sin_grad = df_sin_graduados[df_sin_graduados['es_colombia'] == 1].copy()

    desercion_depto = df_colombia_sin_grad.groupby('departamento').agg({
        '_id': 'count',
        'desertor': 'sum'
    }).reset_index()
    desercion_depto.columns = ['departamento', 'total', 'desertores']
    desercion_depto['tasa_desercion'] = (desercion_depto['desertores'] / desercion_depto['total'] * 100).round(2)

    # Normalizar nombres
    desercion_depto['departamento'] = desercion_depto['departamento'].str.upper().str.strip()
    desercion_depto['departamento'] = desercion_depto['departamento'].replace(mapeo_departamentos)

    # Filtrar departamentos con al menos 10 estudiantes para tasa representativa
    desercion_depto_filtrado = desercion_depto[desercion_depto['total'] >= 10].copy()
    desercion_depto_filtrado = desercion_depto_filtrado.sort_values('tasa_desercion', ascending=True)

    fig_depto_desercion = px.bar(
        desercion_depto_filtrado,
        y='departamento',
        x='tasa_desercion',
        orientation='h',
        text='tasa_desercion',
        labels={'departamento': 'Departamento', 'tasa_desercion': 'Tasa de Deserción (%)'},
        color='tasa_desercion',
        color_continuous_scale='RdYlGn_r'
    )

    fig_depto_desercion.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    fig_depto_desercion.update_layout(
        height=max(500, len(desercion_depto_filtrado) * 20),
        showlegend=False,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_depto_desercion, use_container_width=True)

    st.markdown("---")

    # Box plot de promedio
    st.subheader("Distribución de Promedio Académico")

    fig_promedio_box = px.box(
        df_sin_graduados,
        x='desertor',
        y='promedio',
        color='desertor',
        labels={'desertor': '', 'promedio': 'Promedio Acumulado'},
        color_discrete_map={0: '#00cc96', 1: '#ef553b'}
    )

    fig_promedio_box.update_xaxes(tickvals=[0, 1], ticktext=['No Desertores', 'Desertores'])
    fig_promedio_box.update_layout(showlegend=False, height=500)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig_promedio_box, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        promedio_no_desertor = df_no_desertores['promedio'].mean()
        st.metric("Promedio No Desertores", f"{promedio_no_desertor:.2f}")
    with col2:
        promedio_desertor = df_desertores['promedio'].mean()
        st.metric("Promedio Desertores", f"{promedio_desertor:.2f}")

    st.markdown("---")
    
    # Gráfico de dispersión: Promedio vs ICFES
    st.subheader("Relación: Promedio Acumulado vs Puntaje ICFES")

    # Filtrar valores válidos
    df_scatter = df_sin_graduados[
        (df_sin_graduados['promedio'].notna()) & 
        (df_sin_graduados['puntaje_total'].notna())
    ].copy()

    fig_scatter = px.scatter(
        df_scatter.sample(min(2000, len(df_scatter))),  # Muestra para performance
        x='puntaje_total',
        y='promedio',
        color='desertor',
        labels={
            'puntaje_total': 'Puntaje Total ICFES',
            'promedio': 'Promedio Acumulado',
            'desertor': 'Estado'
        },
        color_discrete_map={0: '#00cc96', 1: '#ef553b'},
        opacity=0.6
    )

    fig_scatter.update_layout(
        height=500,
        legend=dict(
            title='',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemsizing='constant'
        )
    )

    # Cambiar etiquetas de la leyenda
    fig_scatter.for_each_trace(lambda t: t.update(name='No Desertor' if t.name == '0' else 'Desertor'))

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Promedio ICFES por sección
    st.subheader("Promedio ICFES por Sección")

    # Filtro por programa
    programas_disponibles = sorted(df_sin_graduados['programa'].dropna().unique())
    programa_seleccionado = st.selectbox(
        "Seleccionar Programa:",
        ['Todos'] + list(programas_disponibles)
    )

    # Filtrar por programa
    if programa_seleccionado == 'Todos':
        df_icfes = df_sin_graduados.copy()
    else:
        df_icfes = df_sin_graduados[df_sin_graduados['programa'] == programa_seleccionado].copy()

    # Calcular promedios por sección
    secciones_icfes = ['icfes_matematicas', 'icfes_lectura', 'icfes_sociales', 'icfes_ciencias', 'icfes_ingles']
    nombres_secciones = ['Matemáticas', 'Lectura Crítica', 'Sociales', 'Ciencias', 'Inglés']

    promedios_desertores = []
    promedios_no_desertores = []

    for seccion in secciones_icfes:
        prom_deser = df_icfes[df_icfes['desertor'] == 1][seccion].mean()
        prom_no_deser = df_icfes[df_icfes['desertor'] == 0][seccion].mean()
        promedios_desertores.append(prom_deser)
        promedios_no_desertores.append(prom_no_deser)

    df_icfes_prom = pd.DataFrame({
        'Sección': nombres_secciones * 2,
        'Promedio': promedios_no_desertores + promedios_desertores,
        'Tipo': ['No Desertores'] * 5 + ['Desertores'] * 5
    })

    fig_icfes = px.bar(
        df_icfes_prom,
        x='Sección',
        y='Promedio',
        color='Tipo',
        barmode='group',
        text='Promedio',
        color_discrete_map={'Desertores': '#ef553b', 'No Desertores': '#00cc96'}
    )

    fig_icfes.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_icfes.update_layout(
        height=500,
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_icfes, use_container_width=True)

    st.markdown("---")

    # Estudiantes con segundo programa
    st.subheader("Estudiantes con Segundo Programa")

    tiene_segundo = (df_sin_graduados['programa_secundario'].notna()).sum()
    total_sin_grad = len(df_sin_graduados)
    pct_segundo = (tiene_segundo / total_sin_grad * 100)

    # Deserción de estudiantes con segundo programa
    df_con_segundo = df_sin_graduados[df_sin_graduados['programa_secundario'].notna()]
    desertores_con_segundo = df_con_segundo['desertor'].sum()
    tasa_desercion_segundo = (desertores_con_segundo / len(df_con_segundo) * 100) if len(df_con_segundo) > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Estudiantes con 2° Programa", f"{tiene_segundo:,}", 
                  delta=f"{pct_segundo:.2f}%")

    with col2:
        st.metric("Desertores con 2° Programa", f"{desertores_con_segundo:,}")

    with col3:
        st.metric("Tasa de Deserción", f"{tasa_desercion_segundo:.2f}%")

    st.markdown("---")

    # Desertores por tipo de colegio y calendario
    st.subheader("Deserción por Tipo de Colegio y Calendario")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Por Tipo de Colegio")
        
        df_colegio = df_sin_graduados[df_sin_graduados['tipo_colegio'].notna()].copy()
        desercion_colegio = df_colegio.groupby('tipo_colegio').agg({
            '_id': 'count',
            'desertor': 'sum'
        }).reset_index()
        desercion_colegio.columns = ['tipo_colegio', 'total', 'desertores']
        desercion_colegio['tasa_desercion'] = (desercion_colegio['desertores'] / desercion_colegio['total'] * 100).round(2)
        
        fig_colegio = px.bar(
            desercion_colegio,
            x='tipo_colegio',
            y='tasa_desercion',
            text='tasa_desercion',
            labels={'tipo_colegio': 'Tipo de Colegio', 'tasa_desercion': 'Tasa de Deserción (%)'},
            color='tasa_desercion',
            color_continuous_scale='RdYlGn_r'
        )
        
        fig_colegio.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_colegio.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        
        st.plotly_chart(fig_colegio, use_container_width=True)

    with col2:
        st.markdown("##### Por Calendario")
        
        df_calendario = df_sin_graduados[df_sin_graduados['calendario_colegio'].notna()].copy()
        desercion_calendario = df_calendario.groupby('calendario_colegio').agg({
            '_id': 'count',
            'desertor': 'sum'
        }).reset_index()
        desercion_calendario.columns = ['calendario', 'total', 'desertores']
        desercion_calendario['tasa_desercion'] = (desercion_calendario['desertores'] / desercion_calendario['total'] * 100).round(2)
        
        fig_calendario = px.bar(
            desercion_calendario,
            x='calendario',
            y='tasa_desercion',
            text='tasa_desercion',
            labels={'calendario': 'Calendario', 'tasa_desercion': 'Tasa de Deserción (%)'},
            color='tasa_desercion',
            color_continuous_scale='RdYlGn_r'
        )
        
        fig_calendario.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_calendario.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        
        st.plotly_chart(fig_calendario, use_container_width=True)

# ============================================================================
# SECCIÓN 3: MODELO PREDICTIVO
# ============================================================================
else:
    st.title("Modelo Predictivo de Deserción")
    st.markdown("### Predicción de riesgo de deserción estudiantil")
    st.markdown("---")
    
    # Información del modelo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Modelo", "Red Neuronal")
    with col2:
        st.metric("Recall", "85.2%")
    with col3:
        st.metric("Precisión", "67.3%")
    with col4:
        st.metric("AUC", "0.91")
    
    st.markdown("---")
    
    # Predictor Interactivo
    st.subheader("Predictor Interactivo")
    st.markdown("Ingrese los datos del estudiante para predecir el riesgo de deserción:")
    
    # Formulario de entrada
    with st.form("prediction_form"):
        st.markdown("#### Datos Personales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            edad = st.number_input("Edad", min_value=15, max_value=60, value=20)
        with col2:
            genero = st.selectbox("Género", ["Masculino", "Femenino"])
        with col3:
            estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6])
        with col4:
            discapacidad = st.selectbox("Discapacidad", ["No", "Sí"])
        
        st.markdown("#### Información Académica")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            programa = st.selectbox("Programa", sorted(df['programa'].dropna().unique()))
        with col2:
            semestre = st.number_input("Semestre Actual", min_value=1, max_value=15, value=1)
        with col3:
            promedio = st.number_input("Promedio", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            materias_cursadas = st.number_input("Materias Cursadas", min_value=0, max_value=100, value=10)
        with col2:
            materias_perdidas = st.number_input("Materias Perdidas", min_value=0, max_value=50, value=0)
        with col3:
            materias_repetidas = st.number_input("Materias Repetidas", min_value=0, max_value=20, value=0)
        
        st.markdown("#### Puntajes ICFES")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            icfes_mat = st.number_input("Matemáticas", min_value=0, max_value=100, value=50)
        with col2:
            icfes_lec = st.number_input("Lectura", min_value=0, max_value=100, value=50)
        with col3:
            icfes_soc = st.number_input("Sociales", min_value=0, max_value=100, value=50)
        with col4:
            icfes_cie = st.number_input("Ciencias", min_value=0, max_value=100, value=50)
        with col5:
            icfes_ing = st.number_input("Inglés", min_value=0, max_value=100, value=50)
        
        st.markdown("#### Información Adicional")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            becado = st.selectbox("Tipo de Beca", ["No becado", "Institucional", "oficial"])
        with col2:
            tipo_colegio = st.selectbox("Tipo de Colegio", ["OFICIAL", "PRIVADO", "NO APLICA"])
        with col3:
            es_barranquilla = st.selectbox("¿Es de Barranquilla?", ["Sí", "No"])
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir Riesgo de Deserción", use_container_width=True)
        
        if submitted:
            st.markdown("---")
            st.subheader("Resultado de la Predicción")
            
            # Predicción con modelo real de Keras
            if modelo_keras is None:
                st.error("El modelo no está disponible. Por favor, asegúrese de que el archivo 'mejor_modelo_desercion.keras' existe en el directorio.")
                st.stop()
            
            # Predicción con modelo real
            try:
                # Crear datos del estudiante siguiendo EXACTAMENTE el mismo proceso del notebook
                # Paso 1: Crear registro inicial con perdidas_por_depto como diccionario
                datos_estudiante = {
                    'edad': edad,
                    'genero': genero,
                    'estrato': estrato,
                    'discapacidad': discapacidad,
                    'programa': programa,
                    'programa_secundario': 'Ninguno',
                    'tiene_programa_secundario': 0,
                    'semestre_actual': semestre,
                    'tipo_estudiante': 'Pregrado',
                    'tipo_admision': 'Regular',
                    'estado_academico': 'Activo',
                    'ciudad_residencia': 'Barranquilla' if es_barranquilla == "Sí" else 'Otra',
                    'depto_residencia': 'Atlántico' if es_barranquilla == "Sí" else 'Otro',
                    'pais': 'Colombia',
                    'es_barranquilla': 1 if es_barranquilla == "Sí" else 0,
                    'es_colombia': 1,
                    'tipo_colegio': tipo_colegio,
                    'calendario_colegio': 'A',
                    'puntaje_total': icfes_mat + icfes_lec + icfes_soc + icfes_cie + icfes_ing,
                    'matematicas': icfes_mat,
                    'lectura_critica': icfes_lec,  # Nombre correcto del notebook
                    'sociales': icfes_soc,
                    'ciencias': icfes_cie,
                    'ingles': icfes_ing,
                    'promedio': promedio,
                    'materias_cursadas': materias_cursadas,
                    'materias_perdidas': materias_perdidas,
                    'materias_repetidas': materias_repetidas,
                    'perdidas_por_depto': {},  # Diccionario vacío como en el notebook
                    'beca': becado,
                    'ultimo_periodo': '2025-10'
                }
                
                # Paso 2: Crear DataFrame y expandir perdidas_por_depto como en el notebook
                df_pred = pd.DataFrame([datos_estudiante])
                
                # Normalizar perdidas_por_depto y agregar prefix (igual que el notebook)
                perdidas_df = pd.json_normalize(df_pred['perdidas_por_depto'])
                perdidas_df = perdidas_df.add_prefix('perdidas_')
                df_pred = pd.concat([df_pred.drop('perdidas_por_depto', axis=1), perdidas_df], axis=1)
                
                # Obtener TODOS los departamentos únicos de la BD (no solo de un documento)
                all_deptos = set()
                sample_docs = collection.find({'metricas_rendimiento.materias_perdidas_por_departamento': {'$exists': True}}).limit(1000)
                for doc in sample_docs:
                    perdidas_real = doc.get('metricas_rendimiento', {}).get('materias_perdidas_por_departamento', {})
                    if isinstance(perdidas_real, dict):
                        all_deptos.update(perdidas_real.keys())
                
                # Agregar todas las columnas de departamentos que faltan
                for depto in sorted(all_deptos):
                    col_name = f'perdidas_{depto}'
                    if col_name not in df_pred.columns:
                        df_pred[col_name] = 0
                
                # Paso 3: Preprocesar categóricas (mismo orden que el notebook)
                categoricas = ['genero', 'discapacidad', 'programa', 'programa_secundario',
                             'tipo_estudiante', 'tipo_admision', 'estado_academico',
                             'ciudad_residencia', 'depto_residencia', 'pais',
                             'tipo_colegio', 'calendario_colegio', 'ultimo_periodo']
                
                for col in categoricas:
                    if col in df_pred.columns:
                        le = LabelEncoder()
                        df_pred[col] = le.fit_transform(df_pred[col].astype(str))
                
                # Codificar 'beca' también (es categórica)
                if 'beca' in df_pred.columns:
                    le = LabelEncoder()
                    df_pred['beca'] = le.fit_transform(df_pred['beca'].astype(str))
                
                # Paso 4: Escalar datos con StandardScaler
                scaler = StandardScaler()
                X_pred_scaled = scaler.fit_transform(df_pred.values)
                
                # Verificar dimensiones
                if X_pred_scaled.shape[1] != 58:
                    st.warning(f"Dimensiones: {X_pred_scaled.shape[1]} columnas (esperadas: 58)")
                    st.write("Columnas actuales:", list(df_pred.columns))
                
                # Predecir con modelo
                prediccion = modelo_keras.predict(X_pred_scaled, verbose=0)
                probabilidad = float(prediccion[0][0] * 100)
                
                st.success("Predicción realizada con modelo de red neuronal")
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
                st.error("Por favor, contacte al administrador del sistema.")
                st.stop()
            
            # Calcular promedio ICFES para análisis de factores
            puntaje_icfes_promedio = (icfes_mat + icfes_lec + icfes_soc + icfes_cie + icfes_ing) / 5
            
            # Mostrar resultado
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if probabilidad >= 70:
                    st.error(f"### RIESGO ALTO: {probabilidad}%")
                    st.markdown("**Recomendaciones:**")
                    st.markdown("- Intervención inmediata requerida")
                    st.markdown("- Asignar tutor académico")
                    st.markdown("- Evaluar apoyo financiero")
                    st.markdown("- Seguimiento semanal")
                elif probabilidad >= 40:
                    st.warning(f"### RIESGO MEDIO: {probabilidad}%")
                    st.markdown("**Recomendaciones:**")
                    st.markdown("- Monitoreo académico regular")
                    st.markdown("- Apoyo en materias críticas")
                    st.markdown("- Seguimiento quincenal")
                else:
                    st.success(f"### RIESGO BAJO: {probabilidad}%")
                    st.markdown("**Recomendaciones:**")
                    st.markdown("- Continuar con el seguimiento normal")
                    st.markdown("- Mantener rendimiento académico")
                
                # Gráfico de gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probabilidad,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidad de Deserción"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # Factores de riesgo identificados
            st.subheader("Factores de Riesgo Identificados")
            
            factores = []
            if promedio < 3.0:
                factores.append(("Promedio muy bajo", "Alto", f"{promedio:.2f}"))
            elif promedio < 3.5:
                factores.append(("Promedio bajo", "Medio", f"{promedio:.2f}"))
                
            if materias_perdidas > 5:
                factores.append(("Muchas materias perdidas", "Alto", f"{materias_perdidas}"))
            elif materias_perdidas > 2:
                factores.append(("Materias perdidas", "Medio", f"{materias_perdidas}"))
                
            if materias_repetidas > 3:
                factores.append(("Muchas materias repetidas", "Alto", f"{materias_repetidas}"))
            elif materias_repetidas > 0:
                factores.append(("Materias repetidas", "Medio", f"{materias_repetidas}"))
                
            if estrato <= 2:
                factores.append(("Estrato socioeconómico bajo", "Medio", f"Estrato {estrato}"))
                
            if becado == "No becado":
                factores.append(("Sin apoyo financiero", "Medio", "No becado"))
                
            if puntaje_icfes_promedio < 50:
                factores.append(("Puntajes ICFES bajos", "Alto", f"{puntaje_icfes_promedio:.1f}"))
            elif puntaje_icfes_promedio < 60:
                factores.append(("Puntajes ICFES medios", "Bajo", f"{puntaje_icfes_promedio:.1f}"))
            
            if factores:
                df_factores = pd.DataFrame(factores, columns=["Factor", "Nivel", "Valor"])
                
                # Colorear por nivel de riesgo
                def color_nivel(val):
                    if val == "Alto":
                        return 'background-color: #ffcccc'
                    elif val == "Medio":
                        return 'background-color: #fff4cc'
                    else:
                        return 'background-color: #ccffcc'
                
                st.dataframe(
                    df_factores.style.applymap(color_nivel, subset=['Nivel']),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No se identificaron factores de riesgo significativos")
    
    st.markdown("---")
    
    # Mostrar información del modelo
    if info_modelo:
        with st.expander("Información del Modelo de Red Neuronal"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Hiperparámetros:**")
                st.write(f"- ID Modelo: {info_modelo['modelo_id']}")
                st.write(f"- Arquitectura: {info_modelo['hiperparametros']['arquitectura']}")
                st.write(f"- Capas: {info_modelo['hiperparametros']['capas']}")
                st.write(f"- Dropout: {info_modelo['hiperparametros']['dropout']}")
                st.write(f"- Learning Rate: {info_modelo['hiperparametros']['learning_rate']}")
                st.write(f"- Batch Size: {info_modelo['hiperparametros']['batch_size']}")
                st.write(f"- Optimizer: {info_modelo['hiperparametros']['optimizer']}")
            with col2:
                st.write("**Métricas de Desempeño:**")
                st.write(f"- Recall: {info_modelo['metricas']['recall']:.2%}")
                st.write(f"- Precision: {info_modelo['metricas']['precision']:.2%}")
                st.write(f"- F1-Score: {info_modelo['metricas']['f1']:.4f}")
                st.write(f"- AUC: {info_modelo['metricas']['auc']:.4f}")
                st.write(f"- Score Custom: {info_modelo['metricas']['score_custom']:.4f}")
                st.write("**Matriz de Confusión:**")
                st.write(f"- TP: {info_modelo['matriz_confusion']['tp']}, FP: {info_modelo['matriz_confusion']['fp']}")
                st.write(f"- TN: {info_modelo['matriz_confusion']['tn']}, FN: {info_modelo['matriz_confusion']['fn']}")
    else:
        st.info("**Nota:** Este modelo utiliza una red neuronal entrenada con datos históricos de deserción estudiantil.")
