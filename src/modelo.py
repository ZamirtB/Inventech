# Importación de librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import numpy as np

# Cargar los datos generados por datos.py
data = pd.read_csv('data/datos_inversion.csv')

# Filtrar datos para evitar tiempos de recuperación irreales
data = data[data['tiempo_recuperacion'] < 10]  # Filtrar para menos de 10 años

# Preparar los datos para el modelo
X = data[['inversion_inicial', 'ventas', 'rentabilidad', 'gastos_fijos', 'costos_variables', 'flujo_caja', 'crecimiento_ventas']]
y = data['tiempo_recuperacion']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos de entrada
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el escalador para aplicarlo en la predicción
joblib.dump(scaler, 'models/pkl/scaler.pkl')

# Normalizar la variable objetivo usando MinMaxScaler
target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# Guardar el escalador de la variable objetivo
joblib.dump(target_scaler, 'models/pkl/target_scaler.pkl')

# Verificar si el modelo ya existe
if os.path.exists('models/keras/modelo_inversion.keras'):
    # Cargar el modelo existente
    model = load_model('models/keras/modelo_inversion.keras')
    print("Modelo cargado desde 'modelo_inversion.keras'")
    
    # Recompilar el modelo después de cargarlo
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
else:
    # Crear un nuevo modelo si no existe
    model = Sequential()
    model.add(Dense(128, input_dim=7, activation='relu'))  # Capa de entrada y primera capa oculta
    model.add(Dense(64, activation='relu'))                # Segunda capa oculta
    model.add(Dense(32, activation='relu'))                # Tercera capa oculta opcional
    model.add(Dense(16, activation='relu'))                # Cuarta capa oculta opcional
    model.add(Dense(1))                                    # Capa de salida
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    print("Nuevo modelo creado")

# Configurar callbacks para Early Stopping y ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=2000, restore_best_weights=True)
checkpoint = ModelCheckpoint('modelo_inversion_mejor.keras', monitor='val_loss', save_best_only=True)

# Entrenar el modelo, ya sea cargado o nuevo, con 1000 epochs y tamaño de lote mayor
model.fit(X_train_scaled, y_train_scaled, epochs=10000, batch_size=64, validation_data=(X_test_scaled, y_test_scaled), callbacks=[early_stopping, checkpoint])

# Guardar el modelo entrenado (última versión) en el formato recomendado
model.save('models/keras/modelo_inversion.keras')
print("Modelo entrenado y guardado en 'modelo_inversion.keras'")
