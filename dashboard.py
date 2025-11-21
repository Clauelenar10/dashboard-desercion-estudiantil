import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np

st.set_page_config(page_title="Dashboard Deserción", layout="wide")

# Título principal
st.title("Dashboard de Deserción Estudiantil")
st.markdown("### Periodo: 2025-10")
st.markdown("---")

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

# ============================================================================
# SECCIÓN 1: POBLACIÓN GENERAL
# ============================================================================
st.header("Población")

# Métricas principales de población
col1, col2, col3 = st.columns(3)

with col1:
    total_estudiantes = len(df)
    st.metric("Total Estudiantes", f"{total_estudiantes:,}")

with col2:
    total_graduados = df['graduado'].sum()
    st.metric("Graduados", f"{int(total_graduados):,}")

with col3:
    # Total becados (institucional + oficial)
    becados_institucional = (df['becado'] == 'Institucional').sum()
    becados_oficial = (df['becado'] == 'oficial').sum()
    total_becados = becados_institucional + becados_oficial
    st.metric("Total Becados", f"{total_becados:,}")

# Desglose de becados
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("   Becados Institucional", f"{becados_institucional:,}")

with col2:
    st.metric("   Becados Oficial", f"{becados_oficial:,}")

with col3:
    estrato_promedio = df['estrato'].mean()
    st.metric("Estrato Promedio", f"{estrato_promedio:.2f}")

with col4:
    edad_promedio = df['edad'].mean()
    st.metric("Edad Promedio", f"{edad_promedio:.1f} años")

st.markdown("---")

# Ubicación geográfica
col1, col2, col3, col4 = st.columns(4)

with col1:
    estudiantes_barranquilla = (df['es_barranquilla'] == 1).sum()
    pct_barranquilla = (estudiantes_barranquilla / total_estudiantes * 100)
    st.metric("Barranquilla", f"{estudiantes_barranquilla:,}", 
              delta=f"{pct_barranquilla:.1f}%")

with col2:
    estudiantes_no_barranquilla = (df['es_barranquilla'] == 0).sum()
    pct_no_barranquilla = (estudiantes_no_barranquilla / total_estudiantes * 100)
    st.metric("Otras Ciudades", f"{estudiantes_no_barranquilla:,}", 
              delta=f"{pct_no_barranquilla:.1f}%")

with col3:
    estudiantes_colombia = (df['es_colombia'] == 1).sum()
    pct_colombia = (estudiantes_colombia / total_estudiantes * 100)
    st.metric("Colombia", f"{estudiantes_colombia:,}", 
              delta=f"{pct_colombia:.1f}%")

with col4:
    estudiantes_extranjero = (df['es_colombia'] == 0).sum()
    pct_extranjero = (estudiantes_extranjero / total_estudiantes * 100)
    st.metric("Extranjero", f"{estudiantes_extranjero:,}", 
              delta=f"{pct_extranjero:.1f}%")

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

# Mapeo de nombres de departamentos
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

# Diagrama de barras de ciudades de Barranquilla
st.subheader("Estudiantes de Barranquilla por Ciudad")

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
    
    # Ordenar por frecuencia
    estudiantes_ciudad = estudiantes_ciudad.sort_values('total_estudiantes', ascending=True)
    
    # Crear gráfico de barras horizontales
    fig_ciudades = px.bar(
        estudiantes_ciudad,
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
        height=max(400, len(estudiantes_ciudad) * 25),
        showlegend=False,
        xaxis_title="Número de Estudiantes",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_ciudades, use_container_width=True)
else:
    st.warning("No hay datos de estudiantes en Atlántico")

st.markdown("---")

# ============================================================================
# SECCIÓN 3: ANÁLISIS DE DESERCIÓN (SIN GRADUADOS)
# ============================================================================
st.header("Análisis de Deserción")

# Filtrar estudiantes sin graduados
df_sin_graduados = df[df['graduado'] == 0].copy()

# Tasa de deserción general (grande)
tasa_desercion_general = (df_sin_graduados['desertor'].sum() / len(df_sin_graduados) * 100)

st.markdown("### Tasa de Deserción General")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.metric(
        label="",
        value=f"{tasa_desercion_general:.2f}%",
        delta=f"{df_sin_graduados['desertor'].sum():,} de {len(df_sin_graduados):,} estudiantes"
    )

st.markdown("---")

# Deserción por becados
st.subheader("Deserción por Tipo de Beca")

col1, col2, col3 = st.columns(3)

with col1:
    # No becados
    df_no_becados = df_sin_graduados[df_sin_graduados['becado'] == 'No becado']
    if len(df_no_becados) > 0:
        tasa_no_becados = (df_no_becados['desertor'].sum() / len(df_no_becados) * 100)
        st.metric("No Becados", f"{tasa_no_becados:.2f}%", 
                  delta=f"{len(df_no_becados):,} estudiantes")
    else:
        st.metric("No Becados", "N/A")

with col2:
    # Becados institucional
    df_bec_inst = df_sin_graduados[df_sin_graduados['becado'] == 'Institucional']
    if len(df_bec_inst) > 0:
        tasa_bec_inst = (df_bec_inst['desertor'].sum() / len(df_bec_inst) * 100)
        st.metric("Becados Institucional", f"{tasa_bec_inst:.2f}%", 
                  delta=f"{len(df_bec_inst):,} estudiantes")
    else:
        st.metric("Becados Institucional", "N/A")

with col3:
    # Becados oficial
    df_bec_ofi = df_sin_graduados[df_sin_graduados['becado'] == 'oficial']
    if len(df_bec_ofi) > 0:
        tasa_bec_ofi = (df_bec_ofi['desertor'].sum() / len(df_bec_ofi) * 100)
        st.metric("Becados Oficial", f"{tasa_bec_ofi:.2f}%", 
                  delta=f"{len(df_bec_ofi):,} estudiantes")
    else:
        st.metric("🏛️ Becados Oficial", "N/A")

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

st.markdown("---")

# Footer
st.markdown("---")
st.markdown("### Dashboard de Análisis Descriptivo de Deserción Estudiantil")
st.markdown("**Periodo:** 2025-10")
