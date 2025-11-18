
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

# Análisis Geográfico
st.subheader("Análisis Geográfico")

col1, col2 = st.columns(2)

with col1:
    # Mapa de Colombia por departamento
    st.markdown("#### Estudiantes por Departamento (Colombia)")
    
    # Filtrar solo estudiantes de Colombia
    df_colombia = df[df['es_colombia'] == 1].copy()
    
    # Contar por departamento
    estudiantes_depto = df_colombia['departamento'].value_counts().reset_index()
    estudiantes_depto.columns = ['departamento', 'count']
    
    # Calcular deserción por departamento
    desercion_depto = df_colombia.groupby('departamento')['desertor'].agg(['sum', 'count']).reset_index()
    desercion_depto['tasa_desercion'] = (desercion_depto['sum'] / desercion_depto['count'] * 100).round(1)
    desercion_depto.columns = ['departamento', 'desertores', 'total', 'tasa_desercion']
    
    # Merge
    mapa_data = estudiantes_depto.merge(desercion_depto, on='departamento')
    
    # Gráfico de barras horizontales (ya que no tenemos coordenadas exactas)
    fig_mapa = px.bar(mapa_data.sort_values('count', ascending=True).tail(15), 
                      y='departamento', 
                      x='count',
                      orientation='h',
                      title='Top 15 Departamentos',
                      labels={'count': 'Estudiantes', 'departamento': 'Departamento'},
                      color='tasa_desercion',
                      color_continuous_scale='RdYlGn_r',
                      hover_data={'tasa_desercion': ':.1f'})
    
    fig_mapa.update_layout(height=500)
    st.plotly_chart(fig_mapa, use_container_width=True)
    
    # Mostrar tabla de ciudades principales
    st.markdown("##### Principales Ciudades")
    ciudades_top = df_colombia['ciudad'].value_counts().head(10).reset_index()
    ciudades_top.columns = ['Ciudad', 'Estudiantes']
    st.dataframe(ciudades_top, hide_index=True, use_container_width=True)

with col2:
    # Estudiantes internacionales
    st.markdown("#### Estudiantes Internacionales")
    
    # Filtrar estudiantes NO colombianos
    df_internacional = df[df['es_colombia'] == 0].copy()
    
    if len(df_internacional) > 0:
        # Contar por país
        paises = df_internacional['pais'].value_counts().reset_index()
        paises.columns = ['pais', 'count']
        
        # Gráfico de barras
        fig_paises = px.bar(paises, 
                            x='count', 
                            y='pais',
                            orientation='h',
                            title=f'Estudiantes por País ({len(df_internacional)} total)',
                            labels={'count': 'Número de Estudiantes', 'pais': 'País'},
                            color='count',
                            color_continuous_scale='Blues')
        
        fig_paises.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_paises, use_container_width=True)
        
        # Tasa de deserción por país
        st.markdown("##### Deserción por País")
        desercion_pais = df_internacional.groupby('pais').agg({
            'desertor': ['sum', 'count']
        }).reset_index()
        desercion_pais.columns = ['pais', 'desertores', 'total']
        desercion_pais['tasa'] = (desercion_pais['desertores'] / desercion_pais['total'] * 100).round(1)
        desercion_pais = desercion_pais.sort_values('total', ascending=False)
        
        st.dataframe(desercion_pais[['pais', 'total', 'desertores', 'tasa']], 
                     hide_index=True, 
                     use_container_width=True,
                     column_config={
                         'pais': 'País',
                         'total': 'Total',
                         'desertores': 'Desertores',
                         'tasa': st.column_config.NumberColumn('Tasa %', format="%.1f%%")
                     })
    else:
        st.info("No hay estudiantes internacionales en la base de datos")

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