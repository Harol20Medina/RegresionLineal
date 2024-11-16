import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# **Propósito**
# Este programa utiliza un modelo de regresión lineal para analizar y predecir el comportamiento diario
# de los casos confirmados de COVID-19 en un país específico.

# **Justificación**
# La regresión lineal es una técnica estadística adecuada para analizar tendencias en datos temporales
# cuando se espera una relación lineal entre las variables dependientes e independientes. Esto nos permite
# entender y predecir la evolución de los casos diarios.

# **Cargar los datos**
# Justificación: Importamos el archivo CSV que contiene datos acumulados de casos confirmados de COVID-19.
ruta_archivo = r'C:\Users\ben19\Downloads\codigoIA\time_series_covid_19_confirmed.csv'
datos_covid = pd.read_csv(ruta_archivo, delimiter=';')  # Cargamos el archivo con ';' como delimitador.

# **Seleccionar país**
# Propósito: Filtrar y analizar los datos de un país específico.
# Justificación: Permite al usuario elegir el país que desea analizar de forma personalizada.
print("Países disponibles en los datos:")
print(datos_covid['Country/Region'].unique())  # Mostramos los países disponibles en los datos.
pais = input("Ingrese el nombre del país que desea analizar: ").strip()  # Solicitamos al usuario que seleccione un país.

# Validamos que el país ingresado exista en los datos.
if pais not in datos_covid['Country/Region'].values:
    raise ValueError(f"El país '{pais}' no se encuentra en los datos.")  # Lanzamos error si el país no está.

# **Filtrar datos por país**
# Propósito: Extraer los datos específicos del país seleccionado.
# Justificación: Garantiza que el análisis se realice únicamente con datos relevantes del país.
datos_pais = datos_covid[datos_covid['Country/Region'] == pais]

# **Transformar datos en serie de tiempo**
# Propósito: Configurar los datos en formato de serie temporal, con fechas como índice.
# Justificación: Facilita el análisis de tendencias temporales y la implementación de modelos predictivos.
serie_tiempo = datos_pais.iloc[:, 4:-1].T  # Seleccionamos las columnas de fechas y las transponemos.
serie_tiempo.index = pd.to_datetime(serie_tiempo.index, format='%m/%d/%y', errors='coerce')  # Convertimos el índice a fechas.
serie_tiempo.columns = ['Casos Confirmados']  # Renombramos la columna para mayor claridad.

# **Validación del índice**
# Propósito: Asegurar que el índice temporal no tenga problemas como duplicados o valores nulos.
# Justificación: Los índices deben ser únicos y estar ordenados cronológicamente para garantizar resultados correctos.
serie_tiempo = serie_tiempo[~serie_tiempo.index.duplicated(keep='first')]  # Eliminamos fechas duplicadas.
serie_tiempo = serie_tiempo[~serie_tiempo.index.isna()]  # Eliminamos valores nulos en las fechas.
serie_tiempo = serie_tiempo.sort_index()  # Ordenamos las fechas de forma cronológica.
serie_tiempo = serie_tiempo.asfreq('D').ffill()  # Ajustamos la frecuencia a diaria y rellenamos valores faltantes.

# **Calcular casos diarios**
# Propósito: Obtener el número de casos reportados cada día.
# Justificación: Es más informativo analizar cambios diarios que trabajar con datos acumulados.
serie_tiempo['Casos Diarios'] = serie_tiempo['Casos Confirmados'].diff().fillna(0)

# **Definir variables para la regresión**
# Propósito: Crear las variables dependiente e independiente para el modelo de regresión.
# Justificación: La variable independiente (tiempo) explica el comportamiento de la dependiente (casos diarios).
y = serie_tiempo['Casos Diarios'].values  # Variable dependiente: casos diarios.
X = np.arange(len(serie_tiempo)).reshape(-1, 1)  # Variable independiente: tiempo en días como índice numérico.

# **División de datos en entrenamiento y prueba**
# Propósito: Separar los datos en dos conjuntos para ajustar el modelo y evaluar su rendimiento.
# Justificación: Evitar el sobreajuste y medir la capacidad predictiva del modelo en datos no utilizados para el ajuste.
tamano_entrenamiento = int(len(X) * 0.8)  # Calculamos el tamaño del conjunto de entrenamiento.
X_entrenamiento, X_prueba = X[:tamano_entrenamiento], X[tamano_entrenamiento:]  # Dividimos X en entrenamiento y prueba.
y_entrenamiento, y_prueba = y[:tamano_entrenamiento], y[tamano_entrenamiento:]  # Dividimos y en entrenamiento y prueba.

# **Ajustar el modelo de regresión lineal**
# Propósito: Ajustar el modelo con los datos de entrenamiento para encontrar la relación lineal entre las variables.
# Justificación: La regresión lineal estima la pendiente y el intercepto para minimizar los errores.
modelo_lineal = LinearRegression()  # Inicializamos el modelo.
modelo_lineal.fit(X_entrenamiento, y_entrenamiento)  # Ajustamos el modelo con los datos de entrenamiento.

# **Hacer predicciones**
# Propósito: Calcular los valores predichos por el modelo para los conjuntos de entrenamiento y prueba.
# Justificación: Comparar las predicciones con los valores reales permite evaluar la precisión del modelo.
y_pred_entrenamiento = modelo_lineal.predict(X_entrenamiento)  # Predicciones para entrenamiento.
y_pred_prueba = modelo_lineal.predict(X_prueba)  # Predicciones para prueba.

# **Evaluar el modelo**
# Propósito: Medir el rendimiento del modelo utilizando métricas de error.
# Justificación: El MSE evalúa qué tan bien el modelo captura la relación entre las variables.
mse_entrenamiento = mean_squared_error(y_entrenamiento, y_pred_entrenamiento)  # MSE para entrenamiento.
mse_prueba = mean_squared_error(y_prueba, y_pred_prueba)  # MSE para prueba.
print(f"Error Cuadrático Medio en Entrenamiento: {mse_entrenamiento}")
print(f"Error Cuadrático Medio en Prueba: {mse_prueba}")

# **Graficar los resultados de la regresión lineal**
# Propósito: Visualizar las predicciones del modelo frente a los datos reales.
# Justificación: Un gráfico permite interpretar de manera intuitiva el ajuste del modelo.
plt.figure(figsize=(12, 6))  # Configuramos el tamaño del gráfico.
plt.scatter(X, y, color='blue', label='Casos Diarios Reales', alpha=0.6)  # Puntos reales.
plt.plot(X_entrenamiento, y_pred_entrenamiento, color='green', label='Predicción Entrenamiento')  # Línea para entrenamiento.
plt.plot(X_prueba, y_pred_prueba, color='red', linestyle='dashed', label='Predicción Prueba')  # Línea para prueba.
plt.title(f"Regresión Lineal para Casos Diarios de COVID-19 en {pais}")  # Título del gráfico.
plt.xlabel("Días desde el inicio")  # Etiqueta para el eje X.
plt.ylabel("Casos Diarios")  # Etiqueta para el eje Y.
plt.legend()  # Muestra la leyenda.
plt.grid(True)  # Activa la cuadrícula.
plt.show()  # Muestra el gráfico.

# **Graficar residuos**
# Propósito: Verificar si los errores del modelo son aleatorios.
# Justificación: Residuos aleatorios indican un buen ajuste; patrones pueden señalar problemas en el modelo.
residuos_entrenamiento = y_entrenamiento - y_pred_entrenamiento  # Residuos para entrenamiento.
residuos_prueba = y_prueba - y_pred_prueba  # Residuos para prueba.

plt.figure(figsize=(12, 6))  # Configuramos el tamaño del gráfico.
plt.scatter(X_entrenamiento, residuos_entrenamiento, color='green', label='Residuos Entrenamiento')  # Residuos entrenamiento.
plt.scatter(X_prueba, residuos_prueba, color='red', label='Residuos Prueba')  # Residuos prueba.
plt.axhline(0, linestyle='--', color='black', label='Línea Base')  # Línea base en 0.
plt.title(f"Residuos del Modelo de Regresión Lineal para {pais}")  # Título del gráfico.
plt.xlabel("Días desde el inicio")  # Etiqueta para el eje X.
plt.ylabel("Residuos (Casos Reales - Predichos)")  # Etiqueta para el eje Y.
plt.legend()  # Muestra la leyenda.
plt.grid(True)  # Activa la cuadrícula.
plt.show()  # Muestra el gráfico.

# **Exportar resultados**
# Propósito: Guardar los resultados en un archivo CSV para análisis adicional.
# Justificación: Permite realizar análisis posteriores con las predicciones generadas.
serie_tiempo['Predicciones Lineales'] = np.concatenate([y_pred_entrenamiento, y_pred_prueba])  # Agregamos
serie_tiempo.to_csv(r'C:\Users\ben19\Downloads\codigoIA\regresion_lineal_casos_diarios.csv')  # Exportamos los datos.
