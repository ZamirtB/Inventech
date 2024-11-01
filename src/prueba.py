import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado y el escalador
modelo = load_model('models/keras/modelo_inversion.keras')  # Cargar el modelo en formato .keras
scaler = joblib.load('models/pkl/scaler.pkl')  # Cargar el escalador guardado

st.title("Plataforma de Inversiones")
st.write("Ingrese los valores financieros estimados de su proyecto para calcular el tiempo de recuperación.")

# Ingreso de datos del usuario
inversion_inicial = st.number_input('Inversión Inicial', min_value=10000, max_value=10000000, step=1000)
ventas = st.number_input('Ventas Proyectadas', min_value=5000, max_value=5000000, step=1000)
rentabilidad = st.slider('Rentabilidad (%)', min_value=5, max_value=30, step=1) / 100.0
gastos_fijos = st.number_input('Gastos Fijos', min_value=1000, max_value=1000000, step=1000)
costos_variables = st.number_input('Costos Variables', min_value=1000, max_value=1000000, step=1000)
flujo_caja = st.number_input('Flujo de Caja', min_value=2000, max_value=3000000, step=1000)
crecimiento_ventas = st.slider('Crecimiento de Ventas (%)', min_value=1, max_value=20, step=1) / 100.0

# Hacer predicción cuando el usuario presiona el botón
if st.button('Calcular Tiempo de Recuperación'):
    # Crear array con datos de entrada
    datos_usuario = np.array([[inversion_inicial, ventas, rentabilidad, gastos_fijos, costos_variables, flujo_caja, crecimiento_ventas]])
    
    # Aplicar escalador
    datos_usuario_scaled = scaler.transform(datos_usuario)

    # Hacer la predicción
    tiempo_recuperacion = modelo.predict(datos_usuario_scaled)
    st.write(f"El tiempo estimado de recuperación es: {tiempo_recuperacion[0][0]:.2f} años")
