import streamlit as st
import pandas as pd
import joblib
import os

# Activar el modo de página ancha
st.set_page_config(layout="wide")

# Función para agregar la imagen de fondo
def agregar_imagen_fondo():
    import base64
    from pathlib import Path

    image_path = Path("imagenes/fondo.jpg")
    if not image_path.exists():
        st.error("No se encontró la imagen de fondo en la carpeta 'imagenes/'. Verifica la ruta y el nombre del archivo.")
        return

    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
            fondo_css = f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_image});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """
            st.markdown(fondo_css, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al cargar la imagen de fondo: {str(e)}")

# Llamar a la función para añadir la imagen de fondo
agregar_imagen_fondo()

# CSS personalizado para ampliar el ancho de los cuadros y ajustar los márgenes
ancho_css = """
<style>
    .block-container {
        max-width: 95%; /* Ampliar el ancho máximo de la pantalla */
        padding-left: 20px; /* Ajustar margen izquierdo */
        padding-right: 20px; /* Ajustar margen derecho */
    }
</style>
"""
st.markdown(ancho_css, unsafe_allow_html=True)

# Cargar el modelo y los codificadores
modelo = joblib.load('modelo_random_forest.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Ruta absoluta para cargar el archivo CSV
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'datos', 'Tabla_Final.csv')

# Crear columnas con proporciones agresivas [1, 2] para maximizar el ancho de los cuadros
col1, col_space = st.columns([1, 2])

# Cuadro de predicción en la columna izquierda
with col1:
    st.markdown("<h2 style='text-align: left;'>Cuadro de Predicción</h2>", unsafe_allow_html=True)

    provincia = st.selectbox("Selecciona la Provincia", label_encoders['Provincia'].classes_)
    diagnostico = st.selectbox("Selecciona el Diagnóstico", label_encoders['Diagnóstico'].classes_)
    sexo = st.selectbox("Selecciona el Sexo", label_encoders['Sexo'].classes_)
    habitantes = st.number_input("Número de Habitantes", min_value=0)
    metales_pesados = st.number_input("Metales Pesados (As + Cd + Ni + Pb)", min_value=0.0, step=0.0001)
    indice_contaminacion = st.number_input("Índice de Contaminación", min_value=0.0, step=0.1)

    provincia_encoded = label_encoders['Provincia'].transform([provincia])[0]
    diagnostico_encoded = label_encoders['Diagnóstico'].transform([diagnostico])[0]
    sexo_encoded = label_encoders['Sexo'].transform([sexo])[0]

    input_data = pd.DataFrame({
        'Diagnóstico': [diagnostico_encoded],
        'Provincia': [provincia_encoded],
        'Sexo': [sexo_encoded],
        'Habitantes': [habitantes],
        'Metales Pesados': [metales_pesados],
        'Indice_Contaminación': [indice_contaminacion]
    })

    st.markdown("<div style='display: flex; justify-content: center; align-items: center; margin: 10px auto;'>", unsafe_allow_html=True)
    if st.button("Predecir"):
        try:
            # Validar que los campos numéricos no sean cero o negativos
            if habitantes <= 0 or metales_pesados <= 0.0 or indice_contaminacion <= 0.0:
                # Mostrar mensaje de error personalizado con HTML y CSS
                mensaje_error = """
                <div style='background-color: #ffcccc; color: #b30000; padding: 10px; border-radius: 10px; text-align: center; font-weight: bold;'>
                     COMPLETAR TODOS LOS CAMPOS
                </div>
                """
                st.markdown(mensaje_error, unsafe_allow_html=True)
            else:
                # Realizar la predicción
                prediccion = modelo.predict(input_data)
        
                # HTML personalizado para mostrar el resultado
                resultado_html = f"""<div style='display: flex; justify-content: center; align-items: center; background-color: #f0f0f0; width: 50%; height: 50px; padding: 10px; \
	        border-radius: 10px; border: 2px solid #1d741b; margin:20px auto;'> <h2 style='color: #1d741b; text-align: center; font-weight: bold; text-transform: uppercase; font-size: 16px;'>
                PRONÓSTICO DE HOSPITALIZACIONES: {int(prediccion[0])}
                </h2></div>"""
        
            # Mostrar el resultado
            st.markdown(resultado_html, unsafe_allow_html=True)
    
        except Exception:
            # Manejar errores
            pass
# Columna central mínima para separación
with col_space:
    st.markdown("<div style='height: 100%;'></div>", unsafe_allow_html=True)