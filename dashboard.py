
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

total_estudiantes = collection.count_documents({})

# Mostrar en la página
st.metric(label=" Total de Estudiantes", value=total_estudiantes)