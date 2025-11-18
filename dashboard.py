import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
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
        'es_barranquilla': doc['location']['es_barranquilla'],
        'es_colombia': doc['location']['es_colombia'],
        'departamento': doc['location'].get('departamento', ''),
        'pais': doc['location'].get('pais', ''),
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

# Análisis Geográfico
st.subheader("Análisis Geográfico")

# Mapa específico de Atlántico por ciudad
st.markdown("#### Distribución de Estudiantes en Atlántico (por Ciudad)")

# Filtrar estudiantes de Atlántico (antes de normalizar)
df_atlantico = df[df['departamento'].str.upper().str.strip().str.contains('ATLANTICO|ATLÁNTICO', na=False)].copy()

if len(df_atlantico) > 0:
    # Contar por ciudad
    estudiantes_ciudad = df_atlantico.groupby('ciudad').agg({
        'desertor': ['count', 'sum']
    }).reset_index()
    estudiantes_ciudad.columns = ['ciudad', 'total_estudiantes', 'desertores']
    
    # Estandarizar usando z-score (distribución normal)
    mean = estudiantes_ciudad['total_estudiantes'].mean()
    std = estudiantes_ciudad['total_estudiantes'].std()
    estudiantes_ciudad['tamaño_std'] = (estudiantes_ciudad['total_estudiantes'] - mean) / std
    # Escalar a rango positivo (0-100) para el mapa
    min_std = estudiantes_ciudad['tamaño_std'].min()
    max_std = estudiantes_ciudad['tamaño_std'].max()
    estudiantes_ciudad['tamaño'] = ((estudiantes_ciudad['tamaño_std'] - min_std) / (max_std - min_std) * 100).round(0)
    
    estudiantes_ciudad['tasa_desercion'] = (estudiantes_ciudad['desertores'] / estudiantes_ciudad['total_estudiantes'] * 100).round(1)
    
    # Normalizar nombres de ciudades
    estudiantes_ciudad['ciudad'] = estudiantes_ciudad['ciudad'].str.title().str.strip()
    
    # Coordenadas aproximadas de ciudades principales de Atlántico
    coordenadas_atlantico = {
        'Barranquilla': {'lat': 10.9639, 'lon': -74.7964},
        'Soledad': {'lat': 10.9185, 'lon': -74.7694},
        'Malambo': {'lat': 10.8594, 'lon': -74.7739},
        'Sabanalarga': {'lat': 10.6314, 'lon': -74.9222},
        'Puerto Colombia': {'lat': 10.9878, 'lon': -74.9547},
        'Galapa': {'lat': 10.8967, 'lon': -74.8831},
        'Baranoa': {'lat': 10.7942, 'lon': -74.9164},
        'Santo Tomás': {'lat': 10.7503, 'lon': -74.7528},
        'Palmar De Varela': {'lat': 10.7403, 'lon': -74.7542},
        'Sabanagrande': {'lat': 10.7889, 'lon': -74.7617},
        'Juan De Acosta': {'lat': 10.8308, 'lon': -75.0408},
        'Polonuevo': {'lat': 10.7739, 'lon': -74.8528},
        'Usiacurí': {'lat': 10.7372, 'lon': -74.9839},
        'Tubará': {'lat': 10.8833, 'lon': -74.9833},
        'Piojó': {'lat': 10.7622, 'lon': -75.1097},
        'Luruaco': {'lat': 10.6167, 'lon': -75.1500},
        'Repelón': {'lat': 10.4969, 'lon': -75.1333}
}
    # Agregar coordenadas
    estudiantes_ciudad['lat'] = estudiantes_ciudad['ciudad'].map(lambda x: coordenadas_atlantico.get(x, {}).get('lat'))
    estudiantes_ciudad['lon'] = estudiantes_ciudad['ciudad'].map(lambda x: coordenadas_atlantico.get(x, {}).get('lon'))
    
    # Filtrar solo ciudades con coordenadas
    estudiantes_ciudad_map = estudiantes_ciudad[estudiantes_ciudad['lat'].notna()].copy()
    
    # Crear mapa de dispersión
    fig_atlantico = px.scatter_mapbox(
        estudiantes_ciudad_map,
        lat='lat',
        lon='lon',
        size='tamaño',
        color='tasa_desercion',
        hover_name='ciudad',
        hover_data={
            'lat': False,
            'lon': False,
            'total_estudiantes': True,
            'desertores': True,
            'tasa_desercion': ':.1f'
        },
        color_continuous_scale='RdYlGn_r',
        size_max=40,
        zoom=8.5,
        center={'lat': 10.9, 'lon': -74.8},
        mapbox_style='carto-positron',
        labels={
            'total_estudiantes': 'Total',
            'desertores': 'Desertores',
            'tasa_desercion': 'Tasa %'
        }
    )
    
    fig_atlantico.update_layout(
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    st.plotly_chart(fig_atlantico, use_container_width=True)
    
    # Tabla de ciudades
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Top 10 Ciudades")
        top_ciudades = estudiantes_ciudad.sort_values('total_estudiantes', ascending=False).head(10)
        st.dataframe(
            top_ciudades[['ciudad', 'total_estudiantes', 'desertores', 'tasa_desercion']],
            hide_index=True,
            use_container_width=True,
            column_config={
                'ciudad': 'Ciudad',
                'total_estudiantes': 'Total',
                'desertores': 'Desertores',
                'tasa_desercion': st.column_config.NumberColumn('Tasa %', format="%.1f%%")
            }
        )
    
    with col2:
        st.markdown("##### Estadísticas Atlántico")
        st.metric("Total Estudiantes", len(df_atlantico))
        st.metric("Ciudades Representadas", len(estudiantes_ciudad))
        st.metric("Tasa Deserción Promedio", f"{df_atlantico['desertor'].mean() * 100:.1f}%")
else:
    st.warning("No hay datos de estudiantes en Atlántico")

st.markdown("---")



# Mapa de Colombia por departamento
st.markdown("#### Distribución de Estudiantes en Colombia")

# Filtrar solo estudiantes de Colombia
df_colombia = df[df['es_colombia'] == 1].copy()

# Contar por departamento
estudiantes_depto = df_colombia.groupby('departamento').agg({
    'desertor': ['count', 'sum']
}).reset_index()
estudiantes_depto.columns = ['departamento', 'total_estudiantes', 'desertores']
estudiantes_depto['tasa_desercion'] = (estudiantes_depto['desertores'] / estudiantes_depto['total_estudiantes'] * 100).round(1)

# Mapeo de nombres
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

# Separar Atlántico para el mapa (sin Atlántico para mejor escala)
estudiantes_mapa = estudiantes_depto[estudiantes_depto['departamento'] != 'ATLÁNTICO'].copy()

# Cargar GeoJSON
@st.cache_data
def load_geojson():
    url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
    response = requests.get(url)
    return response.json()

geojson_colombia = load_geojson()

# Crear el mapa SIN Atlántico
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
        'total_estudiantes': True,
        'desertores': True,
        'tasa_desercion': ':.1f'
    },
    mapbox_style="carto-positron",
    zoom=4.5,
    center={"lat": 4.5, "lon": -74},
    opacity=0.8,
    labels={'total_estudiantes': 'Total Estudiantes', 
            'tasa_desercion': 'Tasa Deserción %',
            'desertores': 'Desertores'}
)

fig_mapa.update_layout(
    height=600,
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig_mapa, use_container_width=True)

st.info("Nota: Atlántico fue excluido del mapa para mejor visualización de otros departamentos. Ver tabla completa abajo.")

# Tabla resumen CON Atlántico
st.markdown("#### Top 10 Departamentos (Incluye Atlántico)")
top_deptos = estudiantes_depto.sort_values('total_estudiantes', ascending=False).head(10)
st.dataframe(
    top_deptos[['departamento', 'total_estudiantes', 'desertores', 'tasa_desercion']],
    hide_index=True,
    use_container_width=True,
    column_config={
        'departamento': 'Departamento',
        'total_estudiantes': 'Total',
        'desertores': 'Desertores',
        'tasa_desercion': st.column_config.NumberColumn('Tasa %', format="%.1f%%")
    }
)

st.markdown("---")

# Distribución de deserción
st.subheader("Distribución de Deserción")

col1, col2 = st.columns([2, 1])

with col2:
    # Filtros en la derecha
    st.markdown("### Filtros")
    
    # Filtro por estrato (múltiple)
    estratos_disponibles = sorted(df['estrato'].dropna().unique())
    estratos_seleccionados = st.multiselect(
        "Estrato:",
        [int(e) for e in estratos_disponibles],
        default=[int(e) for e in estratos_disponibles]
    )
    
    st.markdown("---")
    
    # Filtro por género (múltiple)
    generos_seleccionados = st.multiselect(
        "Género:",
        ["M", "F"],
        default=["M", "F"]
    )
    
    st.markdown("---")
    
    # Filtro por becado (múltiple)
    becados_seleccionados = st.multiselect(
        "Becado:",
        ["Institucional", "Oficial", "No becado"],
        default=["Institucional", "Oficial", "No becado"]
    )

with col1:
    # Aplicar filtros
    df_filtrado = df.copy()
    
    # Filtrar por estrato
    if len(estratos_seleccionados) > 0:
        df_filtrado = df_filtrado[df_filtrado['estrato'].isin(estratos_seleccionados)]
    
    # Filtrar por género
    if len(generos_seleccionados) > 0:
        df_filtrado = df_filtrado[df_filtrado['genero'].isin(generos_seleccionados)]
    
    # Filtrar por becado
    if len(becados_seleccionados) > 0:
        df_filtrado = df_filtrado[df_filtrado['becado'].isin(becados_seleccionados)]
    
    # Convertir periodo a string y filtrar
    df_filtrado['periodo'] = df_filtrado['periodo'].astype(str)
    df_filtrado = df_filtrado[df_filtrado['periodo'].isin(['202410', '202430', '202510'])]
    
    # Validar que hay datos
    if len(df_filtrado) == 0:
        st.warning("No hay datos para la selección actual")
    else:
        # Contar deserción
        desercion_counts = df_filtrado['desertor'].value_counts()
        
        labels = []
        values = []
        for idx, val in desercion_counts.items():
            labels.append('No Desertor' if idx == 0 else 'Desertor')
            values.append(val)
        
        # Crear gráfica de dona
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=['#00cc96', '#ef553b'],
            texttemplate='%{label}<br>%{value}<br>(%{percent})',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Distribución de Deserción",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas debajo de la gráfica
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Filtrado", len(df_filtrado))
        with col_b:
            desertores = df_filtrado['desertor'].sum()
            st.metric("Desertores", desertores)
        with col_c:
            tasa = (desertores / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
            st.metric("Tasa Deserción", f"{tasa:.1f}%")

st.markdown("---")