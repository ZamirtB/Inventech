import numpy as np
import pandas as pd

# Cantidad de datos a generar
num_datos = 10000

# Generar variables aleatorias en pesos colombianos
inversion_inicial = np.random.randint(10000, 10000000, num_datos)
ventas = np.random.randint(5000, 5000000, num_datos)
rentabilidad = np.random.uniform(0.05, 0.3, num_datos)
gastos_fijos = np.random.randint(1000, 1000000, num_datos)
costos_variables = np.random.randint(1000, 1000000, num_datos)
flujo_caja = np.random.randint(2000, 3000000, num_datos)
crecimiento_ventas = np.random.uniform(0.01, 0.2, num_datos)

# Calcular ganancia neta anual usando una fórmula simplificada
ganancia_neta_anual = (ventas - gastos_fijos - costos_variables) * rentabilidad

# Filtrar valores extremos en ganancia para evitar divisiones por valores pequeños
ganancia_neta_anual[ganancia_neta_anual < 1000] = np.nan

# Calcular tiempo de recuperación solo cuando ganancia_neta_anual es válida
tiempo_recuperacion = np.where(ganancia_neta_anual > 1000, inversion_inicial / ganancia_neta_anual, np.nan)

# Crear DataFrame con los datos generados
data = pd.DataFrame({
    'inversion_inicial': inversion_inicial,
    'ventas': ventas,
    'rentabilidad': rentabilidad,
    'gastos_fijos': gastos_fijos,
    'costos_variables': costos_variables,
    'flujo_caja': flujo_caja,
    'crecimiento_ventas': crecimiento_ventas,
    'ganancia_neta_anual': ganancia_neta_anual,
    'tiempo_recuperacion': tiempo_recuperacion
})

# Guardar los datos generados en un archivo CSV
data.to_csv('data/datos_inversion.csv', index=False)
print("Datos generados y guardados en 'datos_inversion.csv'")
