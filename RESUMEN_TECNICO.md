# Resumen Técnico del Proyecto
## Sistema de Predicción de Deserción Estudiantil

---

## 1. DESCRIPCIÓN DEL PROYECTO

Sistema de análisis predictivo para identificar estudiantes en riesgo de deserción académica, implementado con Machine Learning y visualización interactiva mediante Streamlit Dashboard.

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Base de Datos
- **Plataforma**: Azure Cosmos DB (MongoDB API)
- **Colección**: `Estudiantes.Estudiantes_Materias`
- **Volumen**: 10,226 documentos de estudiantes
- **Estructura**: Documentos JSON con información académica, demográfica y de rendimiento

### 2.2 Pipeline de Datos
```
ESTUDIANTES.xlsx (10,226 registros)
    ↓
Procesamiento y Limpieza (pandas)
    ↓
estudiantes_documentos.json
    ↓
Azure Cosmos DB (MongoDB)
    ↓
Modelos de Machine Learning
    ↓
Dashboard Streamlit
```

---

## 3. MODELOS PREDICTIVOS

### 3.1 Red Neuronal (Modelo Principal)

**Arquitectura:**
- Tipo: Sequential Neural Network
- Capas: medium_4 (configuración óptima)
- Dropout: 0.2
- Framework: TensorFlow/Keras

**Preprocesamiento:**
- SMOTE: 30% oversampling
- StandardScaler para normalización
- Threshold de decisión: 0.35

**Resultados:**
- **Recall**: 76.14% ✓ (cumple requisito ≥ 75%)
- **Precisión**: 17.01%
- **F1 Score**: 27.80%
- **AUC**: 0.809

**Características:**
- Estrategia de entrenamiento: 900 configuraciones evaluadas
- Función de scoring personalizada:
  ```python
  if recall < 0.75:
      score = recall * 0.5  # Penalización
  else:
      score = 0.4 * precision + 0.35 * recall + 0.25 * auc
  ```
- 735 configuraciones lograron recall ≥ 75%

### 3.2 Árbol de Decisión

**Algoritmo:** DecisionTreeClassifier (scikit-learn)

**Resultados:**
- **Recall**: 60.23%
- **Precisión**: 18.40%
- **F1 Score**: 28.19%
- **AUC**: 0.673

**Características:**
- Alta interpretabilidad
- Reglas de decisión claras
- Variables críticas identificadas:
  1. Materias perdidas
  2. Promedio académico
  3. Estrato socioeconómico
  4. Puntaje ICFES
  5. Tipo de beca
  6. Semestre actual

### 3.3 Regresión Logística ⭐ (Mejor Balance)

**Algoritmo:** LogisticRegression (scikit-learn)

**Resultados:**
- **Recall**: 71.59%
- **Precisión**: 18.53% (la más alta)
- **F1 Score**: 29.44% 🏆 (el mejor)
- **AUC**: 0.828 🏆 (el mejor)

**Características:**
- Mejor balance recall-precisión
- Coeficientes interpretables
- Análisis de impacto por variable
- Factores de riesgo positivos: materias perdidas, estrato bajo, sin beca
- Factores protectores: promedio alto, ICFES alto, beca institucional

---

## 4. VARIABLES DEL MODELO

### 4.1 Variables Demográficas
- Edad, género, estrato socioeconómico
- Discapacidad
- Ciudad/departamento de residencia
- Procedencia (Barranquilla vs otras ciudades)

### 4.2 Variables Académicas
- Programa académico
- Semestre actual
- Promedio acumulado
- Materias cursadas, perdidas y repetidas
- Tipo de estudiante y admisión
- Estado académico

### 4.3 Variables de Colegio
- Tipo de colegio (oficial/privado)
- Calendario escolar
- Puntajes ICFES (matemáticas, lectura, sociales, ciencias, inglés)
- Puntaje total ICFES

### 4.4 Variables Financieras
- Tipo de beca (institucional, oficial, ninguna)
- Estrato socioeconómico

---

## 5. ESTRATEGIA DE ENTRENAMIENTO

### 5.1 Requisitos del Modelo
- **Recall mínimo**: 75%
- **Objetivo**: Maximizar precisión sin bajar el recall

### 5.2 Técnicas Aplicadas
- **Oversampling**: SMOTE (30%, 40%, 50%, 60%)
- **Arquitecturas**: light_3, medium_4, deep_3
- **Regularización**: Dropout (0.2, 0.3, 0.4)
- **Thresholds**: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
- **Early Stopping**: Monitoreo de val_loss
- **ReduceLROnPlateau**: Ajuste dinámico de learning rate

### 5.3 Búsqueda de Hiperparámetros
- Configuraciones totales: 900
- Método: Random sampling de 100 configuraciones
- Validación: Train/Val/Test split (60/20/20)
- Métrica de selección: Score custom con penalización por recall < 75%

---

## 6. MÉTRICAS DE RENDIMIENTO

### 6.1 Comparación de Modelos

| Modelo | Recall | AUC | Interpretabilidad | Uso Principal |
|--------|--------|-----|-------------------|---------------|
| Red Neuronal | 76.14% | 0.809 | Baja | Identificación de riesgo (cumple recall ≥75%) |
| Árbol Decisión | 60.23% | 0.673 | Alta | Comprensión de reglas |
| Reg. Logística | 71.59% | 0.828 | Media | Mejor balance general |

### 6.2 Mejoras vs Versión Anterior

| Modelo | Recall Anterior | Recall Actual | Precisión Anterior | Precisión Actual | Mejora Precisión |
|--------|-----------------|---------------|-------------------|------------------|------------------|
| Red Neuronal | 100% | 76.14% | 5.0% | 17.01% | +240% |
| Árbol | 53.5% | 60.23% | 15.0% | 18.40% | +23% |
| Regresión | 69.0% | 71.59% | 17.0% | 18.53% | +9% |

---

## 7. DASHBOARD INTERACTIVO

### 7.1 Tecnología
- **Framework**: Streamlit
- **Visualizaciones**: Plotly, Matplotlib
- **Caché**: TTL de 60 segundos para datos de MongoDB
- **Actualización**: Botón manual de refresh

### 7.2 Secciones del Dashboard

#### A. Características Generales
- Distribución de estudiantes
- Análisis demográfico
- Estadísticas académicas
- Visualizaciones interactivas

#### B. Desertores vs No Desertores
- Comparación de variables
- Análisis de diferencias significativas
- Gráficos comparativos
- Matriz de correlación

#### C. Modelo Predictivo (3 Tabs)
1. **Red Neuronal**
   - Métricas de rendimiento
   - Predictor interactivo
   - Configuración del modelo

2. **Árbol de Decisión**
   - Métricas de rendimiento
   - Reglas de decisión
   - Variables críticas

3. **Regresión Logística**
   - Métricas de rendimiento
   - Coeficientes e interpretación
   - Factores de riesgo/protección

### 7.3 Predictor Interactivo
- Entrada de datos del estudiante
- Selección de modelo (Red Neuronal o Regresión Logística)
- Predicción en tiempo real
- Visualización de probabilidad de deserción
- Factores de riesgo personalizados

---

## 8. INFRAESTRUCTURA TÉCNICA

### 8.1 Lenguajes y Frameworks
- **Python 3.12.11**
- **TensorFlow/Keras**: Modelos de deep learning
- **scikit-learn**: Modelos tradicionales y preprocesamiento
- **pandas**: Manipulación de datos
- **pymongo**: Conexión a MongoDB
- **Streamlit**: Dashboard interactivo
- **Plotly**: Visualizaciones interactivas

### 8.2 Librerías Principales
```python
tensorflow>=2.15.0
scikit-learn>=1.3.0
pandas>=2.0.0
pymongo>=4.5.0
streamlit>=1.28.0
plotly>=5.17.0
imbalanced-learn>=0.11.0  # SMOTE
```

### 8.3 Almacenamiento de Modelos
- **Red Neuronal**: `mejor_modelo_desercion.keras`
- **Árbol de Decisión**: `modelo_arbol_decision.pkl`
- **Regresión Logística**: `modelo_regresion_logistica.pkl`
- **Metadatos**: `mejor_modelo_info.json`

---

## 9. PROCESO DE ACTUALIZACIÓN DE DATOS

### 9.1 Pipeline de Actualización
```bash
1. Actualizar ESTUDIANTES.xlsx y MATERIAS.xlsx
2. Ejecutar DB MONGO.ipynb
   - Procesar datos (10,226 estudiantes)
   - Generar estudiantes_documentos.json
   - Limpiar colección MongoDB
   - Insertar documentos actualizados
3. Ejecutar modelocode.ipynb
   - Entrenar nuevos modelos
   - Evaluar 900 configuraciones
   - Seleccionar mejores modelos
   - Guardar modelos (.keras, .pkl)
4. Actualizar dashboard.py con nuevas métricas
5. Git commit y push a repositorio
```

### 9.2 Tiempo de Ejecución
- **Procesamiento de datos**: ~16 segundos
- **Carga a MongoDB**: ~19 minutos (10,226 docs)
- **Entrenamiento de modelos**: ~30-60 minutos
- **Actualización total**: ~1-2 horas

---

## 10. CONTROL DE VERSIONES

### 10.1 Repositorio
- **Plataforma**: GitHub
- **Owner**: Clauelenar10
- **Repositorio**: dashboard-desercion-estudiantil
- **Branch**: main

### 10.2 Commits Recientes
```
6dccfc8 - Remove examples, advantages/limitations sections, clean emojis, add logistic regression predictor option
6f06b9b - Remove confusion matrix from model details section
6dd209b - Remove Precision and F1 Score from comparison table
3d1ba5c - Add data refresh button and TTL cache for MongoDB data
6a1b812 - Update models with new training data and results (recall >= 75%)
dab7124 - Update model metrics with new training results
```

### 10.3 Archivos Tracked
- `dashboard.py` - Dashboard principal
- `DB MONGO.ipynb` - ETL y carga de datos
- `modelocode.ipynb` - Entrenamiento de modelos
- `ESTUDIANTES.xlsx` - Datos fuente
- `estudiantes_documentos.json` - Datos procesados
- `mejor_modelo_desercion.keras` - Red neuronal
- `modelo_arbol_decision.pkl` - Árbol de decisión
- `modelo_regresion_logistica.pkl` - Regresión logística
- `mejor_modelo_info.json` - Metadatos del modelo
- `requirements.txt` - Dependencias

---

## 11. RESULTADOS Y CONCLUSIONES

### 11.1 Logros Principales
✅ **Recall ≥ 75%**: Cumplimiento del requisito con Red Neuronal (76.14%)  
✅ **Mejor Balance**: Regresión Logística con F1=29.44% y AUC=0.828  
✅ **Mejora en Precisión**: +240% en Red Neuronal vs versión anterior  
✅ **735/900 configuraciones**: Lograron recall ≥ 75%  
✅ **Dashboard Interactivo**: Predictor funcional con 2 modelos  
✅ **Datos Actualizados**: 10,226 estudiantes en Azure Cosmos DB  

### 11.2 Modelo Recomendado por Caso de Uso

| Caso de Uso | Modelo Recomendado | Razón |
|-------------|-------------------|--------|
| Identificar estudiantes en riesgo | Red Neuronal | Cumple recall ≥ 75%, mejor para detección |
| Balance recall-precisión | Regresión Logística | Mejor F1 (29.44%) y AUC (0.828) |
| Entender causas de deserción | Árbol de Decisión | Alta interpretabilidad, reglas claras |
| Análisis de políticas | Regresión Logística | Coeficientes cuantificables |

### 11.3 Limitaciones Identificadas
- **Precisión baja** (~17-18%): Alta tasa de falsos positivos
- **Desbalance de clases**: SMOTE ayuda pero no resuelve completamente
- **Azure Cosmos DB**: Rate limiting (429 errors) en tier gratuito
- **Datos limitados**: 10,226 registros puede limitar generalización

### 11.4 Trabajo Futuro
- [ ] Optimizar precisión sin sacrificar recall
- [ ] Explorar ensemble methods (stacking, voting)
- [ ] Implementar feature engineering avanzado
- [ ] Agregar variables temporales (tendencias por semestre)
- [ ] A/B testing de modelos en producción
- [ ] Migrar a tier pagado de Cosmos DB para mejor rendimiento

---

## 12. CONTACTO Y DOCUMENTACIÓN

**Desarrollador**: Claudia Elena  
**Institución**: Maestría en Big Data  
**Fecha**: Noviembre 2025  
**Versión**: 2.0  

**Documentación adicional**:
- `DB MONGO.ipynb`: Proceso de ETL y carga
- `modelocode.ipynb`: Entrenamiento y evaluación de modelos
- `dashboard.py`: Código del dashboard interactivo
- GitHub: https://github.com/Clauelenar10/dashboard-desercion-estudiantil

---

**Última actualización**: 22 de noviembre de 2025
