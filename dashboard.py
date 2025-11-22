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
    <h1 style="color: white; margin: 0; font-size: 1.8em;">Dashboard</h1>
    <h2 style="color: white; margin: 5px 0 0 0; font-size: 1.2em;">Sistema de Análisis</h2>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("Dashboard Deserción")
st.sidebar.markdown("### Periodo: 2025-10")

# Botón para refrescar datos
if st.sidebar.button("Refrescar Datos", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# Menú de navegación



# Menú de navegación para el resto de secciones

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
            st.warning(f"No se encontró el modelo en {model_path}")
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
        st.warning(f"Error al cargar el modelo: {str(e)}")
        return None, None

modelo_keras, info_modelo = load_keras_model()

# Extraer todos los datos
@st.cache_data(ttl=60)
def load_data(_collection):
    data = list(_collection.find({}))
    return data

datos = load_data(collection)

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

    # Distribución por Género y Edad
    st.subheader("Distribución por Género y Edad")

    col1, col2 = st.columns(2)

    with col1:
        # Distribución por género
        genero_count = df['genero'].value_counts().reset_index()
        genero_count.columns = ['genero', 'count']
        genero_count['porcentaje'] = (genero_count['count'] / genero_count['count'].sum() * 100).round(1)
        
        fig_genero = px.pie(
            genero_count,
            values='count',
            names='genero',
            title='Distribución por Género',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            hole=0.4
        )
        fig_genero.update_traces(textposition='inside', textinfo='percent+label')
        fig_genero.update_layout(height=400)
        st.plotly_chart(fig_genero, use_container_width=True)

    with col2:
        # Distribución por edad
        df_edad = df[df['edad'].notna()].copy()
        df_edad['rango_edad'] = pd.cut(df_edad['edad'], 
                                        bins=[0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 100], 
                                        labels=['Menos de 16', '16', '17', '18', '19', '20', '21', '22', '23', '+24'])
        edad_count = df_edad['rango_edad'].value_counts().reset_index()
        edad_count.columns = ['rango_edad', 'count']
        edad_count = edad_count.sort_values('rango_edad')
        
        fig_edad = px.bar(
            edad_count,
            x='rango_edad',
            y='count',
            title='Distribución por Rango de Edad',
            labels={'rango_edad': 'Rango de Edad', 'count': 'Número de Estudiantes'},
            color='count',
            color_continuous_scale='Blues'
        )
        fig_edad.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_edad, use_container_width=True)

    # Gráfico combinado: Género y Edad
    st.markdown("##### Distribución Combinada: Género por Rango de Edad")
    df_edad_genero = df[(df['edad'].notna()) & (df['genero'].notna())].copy()
    df_edad_genero['rango_edad'] = pd.cut(df_edad_genero['edad'], 
                                          bins=[0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 100], 
                                          labels=['Menos de 16', '16', '17', '18', '19', '20', '21', '22', '23', '+24'])
    edad_genero_count = df_edad_genero.groupby(['rango_edad', 'genero']).size().reset_index(name='count')
    
    fig_edad_genero = px.bar(
        edad_genero_count,
        x='rango_edad',
        y='count',
        color='genero',
        barmode='group',
        labels={'rango_edad': 'Rango de Edad', 'count': 'Número de Estudiantes', 'genero': 'Género'},
        color_discrete_map={'Masculino': '#3498db', 'Femenino': '#e74c3c'}
    )
    fig_edad_genero.update_layout(
        height=450,
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_edad_genero, use_container_width=True)

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
                <h2 style="color: #ff9800; margin: 10px 0 0 0;">{tasa_no_becados:.2f}%</h2>
                <p style="color: #666; margin: 5px 0 0 0;">{len(df_no_becados):,} estudiantes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">No Becados</h3>
                <h2 style="color: #ff9800; margin: 10px 0 0 0;">N/A</h2>
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
                <h2 style="color: #ff9800; margin: 10px 0 0 0;">{tasa_bec_inst:.2f}%</h2>
                <p style="color: #666; margin: 5px 0 0 0;">{len(df_bec_inst):,} estudiantes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">Becados Institucional</h3>
                <h2 style="color: #ff9800; margin: 10px 0 0 0;">N/A</h2>
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
                <h2 style="color: #ff9800; margin: 10px 0 0 0;">{tasa_bec_ofi:.2f}%</h2>
                <p style="color: #666; margin: 5px 0 0 0;">{len(df_bec_ofi):,} estudiantes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #262730; margin: 0;">Becados Oficial</h3>
                <h2 style="color: #ff9800; margin: 10px 0 0 0;">N/A</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Deserción por Género y Edad
    st.subheader("Deserción por Género y Edad")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Tasa de Deserción por Género")
        df_genero = df_sin_graduados[df_sin_graduados['genero'].notna()].copy()
        desercion_genero = df_genero.groupby('genero').agg({
            '_id': 'count',
            'desertor': 'sum'
        }).reset_index()
        desercion_genero.columns = ['genero', 'total', 'desertores']
        desercion_genero['tasa_desercion'] = (desercion_genero['desertores'] / desercion_genero['total'] * 100).round(2)
        
        fig_genero_des = px.bar(
            desercion_genero,
            x='genero',
            y='tasa_desercion',
            text='tasa_desercion',
            labels={'genero': 'Género', 'tasa_desercion': 'Tasa de Deserción (%)'},
            color='genero',
            color_discrete_map={'Masculino': '#3498db', 'Femenino': '#e74c3c'}
        )
        fig_genero_des.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_genero_des.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_genero_des, use_container_width=True)

    with col2:
        st.markdown("##### Tasa de Deserción por Rango de Edad")
        df_edad_des = df_sin_graduados[df_sin_graduados['edad'].notna()].copy()
        df_edad_des['rango_edad'] = pd.cut(df_edad_des['edad'], 
                                           bins=[0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 100], 
                                           labels=['Menos de 16', '16', '17', '18', '19', '20', '21', '22', '23', '+24'])
        desercion_edad = df_edad_des.groupby('rango_edad').agg({
            '_id': 'count',
            'desertor': 'sum'
        }).reset_index()
        desercion_edad.columns = ['rango_edad', 'total', 'desertores']
        desercion_edad['tasa_desercion'] = (desercion_edad['desertores'] / desercion_edad['total'] * 100).round(2)
        desercion_edad = desercion_edad.sort_values('rango_edad')
        
        fig_edad_des = px.bar(
            desercion_edad,
            x='rango_edad',
            y='tasa_desercion',
            text='tasa_desercion',
            labels={'rango_edad': 'Rango de Edad', 'tasa_desercion': 'Tasa de Deserción (%)'},
            color='tasa_desercion',
            color_continuous_scale='Reds'
        )
        fig_edad_des.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_edad_des.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_edad_des, use_container_width=True)

    # Gráfico combinado
    st.markdown("##### Deserción Combinada: Género por Rango de Edad")
    df_edad_genero_des = df_sin_graduados[(df_sin_graduados['edad'].notna()) & (df_sin_graduados['genero'].notna())].copy()
    df_edad_genero_des['rango_edad'] = pd.cut(df_edad_genero_des['edad'], 
                                              bins=[0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 100], 
                                              labels=['Menos de 16', '16', '17', '18', '19', '20', '21', '22', '23', '+24'])
    desercion_edad_genero = df_edad_genero_des.groupby(['rango_edad', 'genero']).agg({
        '_id': 'count',
        'desertor': 'sum'
    }).reset_index()
    desercion_edad_genero.columns = ['rango_edad', 'genero', 'total', 'desertores']
    desercion_edad_genero['tasa_desercion'] = (desercion_edad_genero['desertores'] / desercion_edad_genero['total'] * 100).round(2)
    
    fig_edad_genero_des = px.bar(
        desercion_edad_genero,
        x='rango_edad',
        y='tasa_desercion',
        color='genero',
        barmode='group',
        text='tasa_desercion',
        labels={'rango_edad': 'Rango de Edad', 'tasa_desercion': 'Tasa de Deserción (%)', 'genero': 'Género'},
        color_discrete_map={'Masculino': '#3498db', 'Femenino': '#e74c3c'}
    )
    fig_edad_genero_des.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_edad_genero_des.update_layout(
        height=450,
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_edad_genero_des, use_container_width=True)
    
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
    st.subheader("Distribución de Desertores por Estrato")

    df_desertores = df_sin_graduados[df_sin_graduados['desertor'] == 1]
    df_no_desertores = df_sin_graduados[df_sin_graduados['desertor'] == 0]

    estratos_desertores = df_desertores.groupby('estrato').size().reset_index(name='count')

    fig_estratos = px.bar(
        estratos_desertores,
        x='estrato',
        y='count',
        labels={'estrato': 'Estrato', 'count': 'Número de Desertores'},
        color='count',
        color_continuous_scale='Reds'
    )

    fig_estratos.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        showlegend=False,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_estratos, use_container_width=True)

    st.markdown("---")
    
    # Tasa de deserción por departamento
    st.subheader("Tasa de Deserción por Departamento")

    df_colombia_sin_grad = df_sin_graduados[df_sin_graduados['es_colombia'] == 1].copy()

    # Normalizar nombres primero
    df_colombia_sin_grad['departamento'] = df_colombia_sin_grad['departamento'].str.upper().str.strip()
    df_colombia_sin_grad['departamento'] = df_colombia_sin_grad['departamento'].replace(mapeo_departamentos)
    
    # Unir Cundinamarca con Bogotá
    df_colombia_sin_grad['departamento'] = df_colombia_sin_grad['departamento'].replace('CUNDINAMARCA', 'BOGOTÁ D.C.')

    desercion_depto = df_colombia_sin_grad.groupby('departamento').agg({
        '_id': 'count',
        'desertor': 'sum'
    }).reset_index()
    desercion_depto.columns = ['departamento', 'total', 'desertores']
    desercion_depto['tasa_desercion'] = (desercion_depto['desertores'] / desercion_depto['total'] * 100).round(2)

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
        
        # Asignar colores diferentes a cada tipo de colegio
        colores_colegio = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        desercion_colegio['color'] = [colores_colegio[i % len(colores_colegio)] for i in range(len(desercion_colegio))]
        
        fig_colegio = px.bar(
            desercion_colegio,
            x='tipo_colegio',
            y='tasa_desercion',
            text='tasa_desercion',
            labels={'tipo_colegio': 'Tipo de Colegio', 'tasa_desercion': 'Tasa de Deserción (%)'},
            color='tipo_colegio',
            color_discrete_sequence=colores_colegio
        )
        
        fig_colegio.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_colegio.update_layout(showlegend=False, height=400)
        
        st.plotly_chart(fig_colegio, use_container_width=True)

    with col2:
        st.markdown("##### Por Calendario")
        
        df_calendario = df_sin_graduados[df_sin_graduados['calendario_colegio'].notna()].copy()
        # Filtrar solo calendarios A y B
        df_calendario = df_calendario[df_calendario['calendario_colegio'].isin(['A', 'B'])]
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

    st.markdown("---")

    # Análisis Multivariable
    st.subheader("Análisis Multivariable")
    st.markdown("Exploración de múltiples variables simultáneamente")

    # Crear datos para análisis multivariable
    df_multi = df_sin_graduados[
        (df_sin_graduados['promedio'].notna()) & 
        (df_sin_graduados['puntaje_total'].notna()) &
        (df_sin_graduados['estrato'].notna())
    ].copy()

    # Selector de tipo de gráfico
    tipo_grafico = st.selectbox(
        "Seleccione el tipo de análisis:",
        [
            "Promedio vs ICFES (por Estrato y Deserción)",
            "Promedio vs Materias Perdidas (por Género)",
            "ICFES vs Materias Cursadas (por Tipo de Colegio)",
            "Edad vs Promedio (por Programa)",
            "Matriz de Correlación"
        ]
    )

    if tipo_grafico == "Promedio vs ICFES (por Estrato y Deserción)":
        # Gráfico de burbujas: promedio vs ICFES, tamaño por estrato, color por deserción
        fig_multi = px.scatter(
            df_multi.sample(min(1500, len(df_multi))),
            x='puntaje_total',
            y='promedio',
            size='estrato',
            color='desertor',
            labels={
                'puntaje_total': 'Puntaje Total ICFES',
                'promedio': 'Promedio Acumulado',
                'estrato': 'Estrato',
                'desertor': 'Estado'
            },
            color_discrete_map={0: '#00cc96', 1: '#ef553b'},
            size_max=20,
            opacity=0.6,
            height=600
        )
        fig_multi.for_each_trace(lambda t: t.update(name='No Desertor' if t.name == '0' else 'Desertor'))
        st.plotly_chart(fig_multi, use_container_width=True)

    elif tipo_grafico == "Promedio vs Materias Perdidas (por Género)":
        df_multi_genero = df_multi[df_multi['genero'].notna()].copy()
        fig_multi = px.scatter(
            df_multi_genero,
            x='materias_perdidas',
            y='promedio',
            color='genero',
            facet_col='desertor',
            labels={
                'materias_perdidas': 'Materias Perdidas',
                'promedio': 'Promedio Acumulado',
                'genero': 'Género',
                'desertor': 'Estado'
            },
            color_discrete_map={'Masculino': '#3498db', 'Femenino': '#e74c3c'},
            opacity=0.6,
            height=500
        )
        fig_multi.for_each_annotation(lambda a: a.update(text='No Desertor' if a.text.split('=')[1] == '0' else 'Desertor'))
        st.plotly_chart(fig_multi, use_container_width=True)

    elif tipo_grafico == "ICFES vs Materias Cursadas (por Tipo de Colegio)":
        df_multi_colegio = df_multi[df_multi['tipo_colegio'].notna()].copy()
        df_multi_colegio = df_multi_colegio[df_multi_colegio['materias_cursadas'] > 0]
        fig_multi = px.scatter(
            df_multi_colegio.sample(min(1500, len(df_multi_colegio))),
            x='materias_cursadas',
            y='puntaje_total',
            color='tipo_colegio',
            symbol='desertor',
            labels={
                'materias_cursadas': 'Materias Cursadas',
                'puntaje_total': 'Puntaje ICFES',
                'tipo_colegio': 'Tipo de Colegio',
                'desertor': 'Estado'
            },
            opacity=0.6,
            height=600
        )
        fig_multi.for_each_trace(lambda t: t.update(name=t.name.replace(', 0', ' - No Desertor').replace(', 1', ' - Desertor')))
        st.plotly_chart(fig_multi, use_container_width=True)

    elif tipo_grafico == "Edad vs Promedio (por Programa)":
        # Seleccionar top 5 programas por cantidad de estudiantes
        top_programas = df_multi['programa'].value_counts().head(5).index.tolist()
        df_multi_prog = df_multi[df_multi['programa'].isin(top_programas)].copy()
        
        fig_multi = px.box(
            df_multi_prog,
            x='programa',
            y='promedio',
            color='desertor',
            labels={
                'programa': 'Programa',
                'promedio': 'Promedio Acumulado',
                'desertor': 'Estado'
            },
            color_discrete_map={0: '#00cc96', 1: '#ef553b'},
            height=600
        )
        fig_multi.for_each_trace(lambda t: t.update(name='No Desertor' if t.name == '0' else 'Desertor'))
        fig_multi.update_xaxes(tickangle=45)
        st.plotly_chart(fig_multi, use_container_width=True)

    else:  # Matriz de Correlación
        # Seleccionar variables numéricas relevantes
        variables_numericas = [
            'edad', 'estrato', 'promedio', 'puntaje_total',
            'materias_cursadas', 'materias_perdidas', 'materias_repetidas',
            'icfes_matematicas', 'icfes_lectura', 'desertor'
        ]
        
        df_corr = df_multi[variables_numericas].dropna()
        matriz_corr = df_corr.corr()
        
        fig_multi = px.imshow(
            matriz_corr,
            labels=dict(x="Variable", y="Variable", color="Correlación"),
            x=matriz_corr.columns,
            y=matriz_corr.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto='.2f',
            height=700
        )
        fig_multi.update_layout(
            title="Matriz de Correlación entre Variables",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_multi, use_container_width=True)
        
        st.info("Valores cercanos a 1 indican correlación positiva fuerte, cercanos a -1 correlación negativa fuerte, y cercanos a 0 poca o ninguna correlación.")

# ============================================================================
# SECCIÓN 3: MODELO PREDICTIVO
# ============================================================================
else:
    st.title("Modelo Predictivo de Deserción")
    st.markdown("### Predicción de riesgo de deserción estudiantil")
    st.markdown("---")
    
    # Tabs para diferentes modelos
    tab1, tab2, tab3 = st.tabs(["Red Neuronal (Principal)", "Árbol de Decisión", "Regresión Logística"])
    
    # ========== TAB 1: RED NEURONAL ==========
    with tab1:
        st.subheader("Modelo de Red Neuronal - Mejor Desempeño")
        
        # Información del modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recall", "75.00%", help="Detecta 3 de cada 4 estudiantes en riesgo")
        with col2:
            st.metric("AUC", "0.814")
        
        st.info("**Modelo optimizado con recall ≥ 75%**: Balance entre detectar estudiantes en riesgo (75% recall) y mantener precisión aceptable (16.84%). Configuración modelo_v2_48.")
        
        st.markdown("---")
        
        # Predictor Interactivo
        st.subheader("Predictor Interactivo")
        st.markdown("Ingrese los datos del estudiante para predecir el riesgo de deserción:")
        
        # Selector de modelo
        modelo_seleccionado = st.radio(
            "Seleccione el modelo para predicción:",
            ["Red Neuronal", "Regresión Logística"],
            horizontal=True
        )
    
    # Inicializar timestamp para keys únicos
    import time
    if 'form_key' not in st.session_state:
        st.session_state.form_key = 0
    
    # Formulario de entrada
    with st.form(key=f"prediction_form_{st.session_state.form_key}"):
        st.markdown("#### Datos Personales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            edad = st.number_input("Edad", min_value=15, max_value=60, value=20, key=f"edad_{st.session_state.form_key}")
        with col2:
            genero = st.selectbox("Género", ["Masculino", "Femenino"], key=f"genero_{st.session_state.form_key}")
        with col3:
            estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6], key=f"estrato_{st.session_state.form_key}")
        with col4:
            discapacidad = st.selectbox("Discapacidad", ["No", "Sí"], key=f"discapacidad_{st.session_state.form_key}")
        
        st.markdown("#### Información Académica")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            programa = st.selectbox("Programa", sorted(df['programa'].dropna().unique()), key=f"programa_{st.session_state.form_key}")
        with col2:
            semestre = st.number_input("Semestre Actual", min_value=1, max_value=15, value=1, key=f"semestre_{st.session_state.form_key}")
        with col3:
            promedio = st.number_input("Promedio", min_value=0.0, max_value=5.0, value=3.5, step=0.1, key=f"promedio_{st.session_state.form_key}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            materias_cursadas = st.number_input("Materias Cursadas", min_value=0, max_value=100, value=10, key=f"cursadas_{st.session_state.form_key}")
        with col2:
            materias_perdidas = st.number_input("Materias Perdidas", min_value=0, max_value=50, value=0, key=f"perdidas_{st.session_state.form_key}")
        with col3:
            materias_repetidas = st.number_input("Materias Repetidas", min_value=0, max_value=20, value=0, key=f"repetidas_{st.session_state.form_key}")
        
        st.markdown("#### Puntajes ICFES")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            icfes_mat = st.number_input("Matemáticas", min_value=0, max_value=100, value=50, key=f"mat_{st.session_state.form_key}")
        with col2:
            icfes_lec = st.number_input("Lectura", min_value=0, max_value=100, value=50, key=f"lec_{st.session_state.form_key}")
        with col3:
            icfes_soc = st.number_input("Sociales", min_value=0, max_value=100, value=50, key=f"soc_{st.session_state.form_key}")
        with col4:
            icfes_cie = st.number_input("Ciencias", min_value=0, max_value=100, value=50, key=f"cie_{st.session_state.form_key}")
        with col5:
            icfes_ing = st.number_input("Inglés", min_value=0, max_value=100, value=50, key=f"ing_{st.session_state.form_key}")
        
        st.markdown("#### Información Adicional")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            becado = st.selectbox("Tipo de Beca", ["No becado", "Institucional", "oficial"], key=f"beca_{st.session_state.form_key}")
        with col2:
            tipo_colegio = st.selectbox("Tipo de Colegio", ["OFICIAL", "PRIVADO", "NO APLICA"], key=f"colegio_{st.session_state.form_key}")
        with col3:
            es_barranquilla = st.selectbox("¿Es de Barranquilla?", ["Sí", "No"], key=f"barranquilla_{st.session_state.form_key}")
        
        # Botones de acción
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submitted = st.form_submit_button("Predecir Riesgo de Deserción", use_container_width=True, type="primary")
        with col_btn2:
            limpiar = st.form_submit_button("Limpiar Formulario", use_container_width=True)
    
    # Manejar limpiar formulario
    if limpiar:
        st.session_state.form_key += 1
        st.rerun()
    
    # Manejar predicción
    if submitted:
            # Forzar limpieza de cache en cada predicción
            import hashlib
            input_hash = hashlib.md5(f"{edad}{genero}{estrato}{programa}{semestre}{promedio}{materias_perdidas}".encode()).hexdigest()
            
            st.markdown("---")
            st.subheader(f"Resultado de la Predicción")
            
            # Predicción con modelo real de Keras
            if modelo_keras is None:
                st.error("El modelo no está disponible. Por favor, asegúrese de que el archivo 'mejor_modelo_desercion.keras' existe en el directorio.")
                st.stop()
            
            # Predicción con modelo real
            try:
                # PASO 1: Preparar datos de entrenamiento para scaler (usar datos históricos)
                # Cargar todos los datos de MongoDB para entrenar scaler correctamente
                datos_training = list(collection.find({}).limit(5000))
                
                # Crear DataFrame de entrenamiento
                training_records = []
                for doc in datos_training:
                    record = {
                        'edad': doc['datos_personales'].get('edad'),
                        'genero': doc['datos_personales'].get('genero', ''),
                        'estrato': doc['datos_personales'].get('estrato'),
                        'discapacidad': doc['datos_personales'].get('discapacidad', ''),
                        'programa': doc['academico'].get('programa', ''),
                        'programa_secundario': doc['academico'].get('programa_secundario', 'Ninguno'),
                        'tiene_programa_secundario': 1 if doc['academico'].get('programa_secundario') not in [None, 'Ninguno', ''] else 0,
                        'semestre_actual': doc['academico'].get('semestre_actual'),
                        'tipo_estudiante': doc['academico'].get('tipo_estudiante', ''),
                        'tipo_admision': doc['academico'].get('tipo_admision', ''),
                        'estado_academico': doc['academico'].get('estado_academico', ''),
                        'ciudad_residencia': doc['location'].get('ciudad', ''),
                        'depto_residencia': doc['location'].get('departamento', ''),
                        'pais': doc['location'].get('pais', ''),
                        'es_barranquilla': doc['location'].get('es_barranquilla', 0),
                        'es_colombia': doc['location'].get('es_colombia', 0),
                        'tipo_colegio': doc['colegio'].get('tipo_colegio'),
                        'calendario_colegio': doc['colegio'].get('calendario_colegio'),
                        'puntaje_total': doc['ICFES'].get('puntaje_total'),
                        'matematicas': doc['ICFES'].get('matematicas'),
                        'lectura_critica': doc['ICFES'].get('lectura_critica'),
                        'sociales': doc['ICFES'].get('sociales'),
                        'ciencias': doc['ICFES'].get('ciencias'),
                        'ingles': doc['ICFES'].get('ingles'),
                        'promedio': doc['metricas_rendimiento'].get('promedio_acumulado'),
                        'materias_cursadas': doc['metricas_rendimiento'].get('materias_cursadas_total', 0),
                        'materias_perdidas': doc['metricas_rendimiento'].get('materias_perdidas_total', 0),
                        'materias_repetidas': doc['metricas_rendimiento'].get('materias_repetidas', 0),
                        'perdidas_por_depto': doc['metricas_rendimiento'].get('materias_perdidas_por_departamento', {}),
                        'beca': doc['estado'].get('becado', ''),
                        'ultimo_periodo': doc.get('ultimo_periodo', 202510)
                    }
                    training_records.append(record)
                
                df_training = pd.DataFrame(training_records)
                
                # PASO 2: Crear datos del estudiante a predecir
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
                
                # PASO 3: Preparar datos de training expandiendo perdidas_por_depto
                perdidas_df_train = pd.json_normalize(df_training['perdidas_por_depto'])
                perdidas_df_train = perdidas_df_train.add_prefix('perdidas_')
                df_training = pd.concat([df_training.drop('perdidas_por_depto', axis=1), perdidas_df_train], axis=1)
                
                # PASO 4: Preparar datos del estudiante a predecir
                df_pred = pd.DataFrame([datos_estudiante])
                perdidas_df_pred = pd.json_normalize(df_pred['perdidas_por_depto'])
                perdidas_df_pred = perdidas_df_pred.add_prefix('perdidas_')
                df_pred = pd.concat([df_pred.drop('perdidas_por_depto', axis=1), perdidas_df_pred], axis=1)
                
                # PASO 5: Asegurar que ambos DataFrames tengan las mismas columnas
                # Agregar columnas faltantes en df_pred
                for col in df_training.columns:
                    if col not in df_pred.columns:
                        df_pred[col] = 0
                
                # Reordenar columnas de df_pred para que coincidan con df_training
                df_pred = df_pred[df_training.columns]
                
                # PASO 6: Preprocesar categóricas usando TODOS los datos (training + pred)
                categoricas = ['genero', 'discapacidad', 'programa', 'programa_secundario',
                             'tipo_estudiante', 'tipo_admision', 'estado_academico',
                             'ciudad_residencia', 'depto_residencia', 'pais',
                             'tipo_colegio', 'calendario_colegio', 'beca', 'ultimo_periodo']
                
                # Combinar training y predicción temporalmente para fit de LabelEncoders
                df_combined = pd.concat([df_training, df_pred], ignore_index=True)
                encoders = {}
                
                for col in categoricas:
                    if col in df_combined.columns:
                        le = LabelEncoder()
                        df_combined[col] = le.fit_transform(df_combined[col].astype(str))
                        encoders[col] = le
                
                # Separar de nuevo
                df_training_encoded = df_combined.iloc[:-1]
                df_pred_encoded = df_combined.iloc[-1:]
                
                # PASO 7: Escalar datos usando StandardScaler FIT en training, TRANSFORM en predicción
                scaler = StandardScaler()
                scaler.fit(df_training_encoded.values)
                X_pred_scaled = scaler.transform(df_pred_encoded.values)
                
                # Verificar dimensiones
                if X_pred_scaled.shape[1] != 58:
                    st.warning(f"Dimensiones: {X_pred_scaled.shape[1]} columnas (esperadas: 58)")
                    st.write("Columnas actuales:", list(df_pred.columns))
                
                # DEBUG: Mostrar datos de entrada
                with st.expander("🔍 Ver datos de entrada (debug)"):
                    st.write("**Datos del estudiante:**")
                    st.write(f"- Edad: {edad}, Género: {genero}, Estrato: {estrato}")
                    st.write(f"- Programa: {programa}, Semestre: {semestre}")
                    st.write(f"- Promedio: {promedio}, Materias perdidas: {materias_perdidas}")
                    st.write(f"- ICFES: Mat={icfes_mat}, Lec={icfes_lec}")
                    st.write(f"**Shape de entrada al modelo:** {X_pred_scaled.shape}")
                    st.write(f"**Muestra de datos escalados (primeros 10):** {X_pred_scaled[0][:10]}")
                
                # Predecir con modelo
                prediccion = modelo_keras.predict(X_pred_scaled, verbose=0)
                probabilidad = float(prediccion[0][0] * 100)
                
                st.success(f"Predicción realizada con modelo de red neuronal")
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
                st.error("Por favor, contacte al administrador del sistema.")
                st.stop()
            
            # Calcular promedio ICFES para análisis de factores
            puntaje_icfes_promedio = (icfes_mat + icfes_lec + icfes_soc + icfes_cie + icfes_ing) / 5
            
            # Mostrar resultado
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Mostrar resultado como Desertor/No Desertor según puntaje redondeado
                # Solo si el puntaje redondeado es exactamente 100 es desertor
                # Solo si el puntaje es mayor o igual a 99.96 es desertor
                if probabilidad >= 99.96:
                    st.error("### DESERTOR")
                else:
                    st.success("### NO DESERTOR")
                
                # No mostrar gráfico, solo resultado
            
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
    else:
        st.info("**Nota:** Este modelo utiliza una red neuronal entrenada con datos históricos de deserción estudiantil.")
    
    # ========== TAB 2: ÁRBOL DE DECISIÓN ==========
    with tab2:
        st.subheader("Modelo de Árbol de Decisión")
        st.markdown("Modelo interpretable que muestra reglas de decisión claras")
        
        # Métricas del árbol de decisión (valores del notebook)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recall", "60.23%", help="Detecta 6 de cada 10 estudiantes en riesgo")
        with col2:
            st.metric("AUC", "0.673")
        
        st.warning("**Nota**: Este modelo ofrece alta interpretabilidad con reglas claras. Precisión similar a Red Neuronal pero menor recall.")
        
        st.markdown("---")
        
        st.subheader("Reglas de Decisión Principales")
        st.markdown("""
        El árbol de decisión utiliza las siguientes variables clave para predecir deserción:
        
        **Variables Más Importantes:**
        1. **Promedio académico**: Estudiantes con promedio < 3.0 tienen mayor riesgo
        2. **Materias perdidas**: Más de 3 materias perdidas indica alto riesgo
        3. **Puntaje ICFES total**: Puntajes < 200 están asociados con deserción
        4. **Materias repetidas**: Repetir materias aumenta significativamente el riesgo
        5. **Estrato socioeconómico**: Estratos 1-2 muestran mayor vulnerabilidad
        6. **Semestre actual**: Mayor riesgo en semestres iniciales (1-3)
        """)
    
    # ========== TAB 3: REGRESIÓN LOGÍSTICA ==========
    with tab3:
        st.subheader("Modelo de Regresión Logística")
        st.markdown("Modelo lineal que muestra el impacto individual de cada variable")
        
        # Métricas de regresión logística (valores del notebook)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recall", "71.59%", help="Detecta 7 de cada 10 estudiantes en riesgo")
        with col2:
            st.metric("AUC", "0.828")
        
        st.warning("**Nota**: Mejor F1 Score (29.44%) y AUC (0.828) entre todos los modelos. Excelente balance recall-precisión.")
        
        st.markdown("---")
        
        st.subheader("Coeficientes e Interpretación")
        st.markdown("""
        La regresión logística asigna un **peso (coeficiente)** a cada variable:
        - **Coeficiente positivo** → Aumenta la probabilidad de deserción
        - **Coeficiente negativo** → Disminuye la probabilidad de deserción
        
        **Principales Factores que AUMENTAN el riesgo (+):**
        1. **Materias perdidas** (+0.15 a +0.25 por materia)
        2. **Materias repetidas** (+0.10 a +0.20 por materia)
        3. **Estrato bajo** (+0.30 a +0.50 para estratos 1-2)
        4. **Edad mayor** (+0.05 por año adicional)
        5. **No tener beca** (+0.20 a +0.40)
        
        **Principales Factores que REDUCEN el riesgo (-):**
        1. **Promedio alto** (-0.40 a -0.60 por punto de promedio)
        2. **Puntaje ICFES alto** (-0.30 a -0.50)
        3. **Tener beca institucional** (-0.30 a -0.50)
        4. **Semestre avanzado** (-0.10 por semestre)
        5. **Colegio privado** (-0.15 a -0.25)
        """)
        

    
    st.markdown("---")
    st.markdown("### Comparación de Modelos")
    
    # Tabla comparativa
    comparacion_modelos = pd.DataFrame({
        'Modelo': ['Red Neuronal', 'Árbol de Decisión', 'Regresión Logística'],
        'Recall': ['76.14%', '60.23%', '71.59%'],
        'AUC': ['0.809', '0.673', '0.828'],
        'Interpretabilidad': ['Baja', 'Alta', 'Media'],
        'Uso Recomendado': [
            'Recall ≥75% con mejor precisión',
            'Entender reglas de decisión claras',
            'Mejor balance general (F1 y AUC)'
        ]
    })
    
    st.dataframe(comparacion_modelos, use_container_width=True, hide_index=True)
    
    st.success("**Conclusión**: La Regresión Logística ofrece el mejor balance (F1: 29.44%, AUC: 0.828). La Red Neuronal cumple requisito recall ≥75% con mejor precisión. Árbol de Decisión aporta interpretabilidad.")

