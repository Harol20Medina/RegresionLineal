import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# **Propósito**
# Este programa utiliza un modelo de regresión lineal para analizar y predecir los casos diarios de COVID-19 en un país específico.
# Además, incluye gráficos explicativos para visualizar la predicción y evaluar el rendimiento del modelo.

# **Cargar los datos**
# Justificación: Importamos los datos acumulados de casos confirmados de COVID-19 desde un archivo CSV.
ruta_archivo = r'C:\Users\ben19\Downloads\codigoIA\time_series_covid_19_confirmed.csv'
datos_covid = pd.read_csv(ruta_archivo, delimiter=';')  # Cargamos el archivo con el delimitador ';'.

# **Seleccionar país**
# Justificación: Permitimos al usuario elegir el país que desea analizar.
print("Países disponibles en los datos:")
print(datos_covid['Country/Region'].unique())  # Mostramos los países disponibles en los datos.
pais = input("Ingrese el nombre del país que desea analizar: ").strip()  # Solicitamos el país al usuario.

# Validación del país ingresado
if pais not in datos_covid['Country/Region'].values:
    raise ValueError(f"El país '{pais}' no se encuentra en los datos.")  # Error si el país no está en los datos.

# **Filtrar datos por país**
# Propósito: Extraemos solo las filas correspondientes al país seleccionado.
# Justificación: Nos enfocamos en un conjunto de datos relevante para el análisis.
datos_pais = datos_covid[datos_covid['Country/Region'] == pais]

# **Transformar datos en serie de tiempo**
# Propósito: Configurar los datos en formato de serie temporal, con las fechas como índice.
# Justificación: El análisis de series temporales requiere un índice temporal para detectar tendencias.
serie_tiempo = datos_pais.iloc[:, 4:-1].T  # Seleccionamos columnas de fechas y las transponemos.
serie_tiempo.index = pd.to_datetime(serie_tiempo.index, format='%m/%d/%y', errors='coerce')  # Convertimos a fechas.
serie_tiempo.columns = ['Casos Confirmados']  # Renombramos la columna para mayor claridad.

# Validación del índice
# Propósito: Garantizar que el índice no tenga duplicados y esté ordenado cronológicamente.
serie_tiempo = serie_tiempo[~serie_tiempo.index.duplicated(keep='first')]  # Eliminamos duplicados.
serie_tiempo = serie_tiempo.sort_index()  # Ordenamos cronológicamente.
serie_tiempo = serie_tiempo.asfreq('D').ffill()  # Ajustamos la frecuencia diaria y rellenamos valores faltantes.

# **Calcular casos diarios**
# Propósito: Obtener el número de casos reportados por día en lugar de acumulados.
# Justificación: Analizar los casos diarios es más informativo para detectar cambios diarios en los contagios.
serie_tiempo['Casos Diarios'] = serie_tiempo['Casos Confirmados'].diff().fillna(0)

# **Definir variables para la regresión**
# Propósito: Crear las variables dependiente (casos diarios) e independiente (tiempo en días).
# Justificación: Estas variables son esenciales para ajustar el modelo de regresión lineal.
y = serie_tiempo['Casos Diarios'].values  # Variable dependiente: casos diarios.
X = np.arange(len(serie_tiempo)).reshape(-1, 1)  # Variable independiente: días como índice numérico.

# **Ajustar el modelo de regresión lineal**
# Propósito: Ajustar un modelo lineal que relacione el tiempo con los casos diarios.
# Justificación: La regresión lineal modela una relación entre las variables independiente y dependiente.
modelo_lineal = LinearRegression()
modelo_lineal.fit(X, y)  # Ajustamos el modelo con los datos.

# **Predicciones**
# Propósito: Generar predicciones de casos diarios utilizando el modelo ajustado.
# Justificación: Comparar las predicciones con los valores reales permite evaluar el modelo.
serie_tiempo['Predicciones'] = modelo_lineal.predict(X)

# **Métricas de evaluación**
# Propósito: Calcular métricas para evaluar el rendimiento del modelo.
# Justificación: El MSE, MAE y R² son métricas estándar para medir el ajuste de modelos de regresión.
mse = mean_squared_error(y, serie_tiempo['Predicciones'])  # Error cuadrático medio.
mae = mean_absolute_error(y, serie_tiempo['Predicciones'])  # Error absoluto medio.
r2 = r2_score(y, serie_tiempo['Predicciones'])  # Coeficiente de determinación.

# Mostrar métricas en la consola
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")

# **Gráfico 1: Casos reales y predicciones**
# Propósito: Comparar los casos reales con la línea de regresión predicha.
plt.figure(figsize=(14, 7))
plt.scatter(serie_tiempo.index, serie_tiempo['Casos Diarios'], color='blue', alpha=0.6, label='Casos Reales (Diarios)')
plt.plot(serie_tiempo.index, serie_tiempo['Predicciones'], color='red', linewidth=2, label='Línea de Regresión')
plt.title(f"Casos Diarios vs Predicción para COVID-19 en {pais}", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Casos Diarios", fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
plt.show()

# **Gráfico 2: Residuos**
# Propósito: Evaluar si los errores (residuos) son aleatorios.
# Justificación: Residuos aleatorios indican que el modelo captura correctamente la tendencia.
residuos = y - serie_tiempo['Predicciones']
plt.figure(figsize=(14, 7))
plt.scatter(serie_tiempo.index, residuos, color='green', alpha=0.6, label='Residuos (Errores)')
plt.axhline(0, color='black', linestyle='--', label='Línea Base')
plt.title(f"Residuos del Modelo de Regresión Lineal para {pais}", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Residuos (Casos Reales - Predichos)", fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
plt.show()

# **Gráfico 3: Predicción Acumulada**
# Propósito: Visualizar cómo se acumulan los casos reales y predichos a lo largo del tiempo.
# Justificación: Permite analizar tendencias generales a largo plazo.
serie_tiempo['Casos Predichos Acumulados'] = serie_tiempo['Predicciones'].cumsum()
serie_tiempo['Casos Reales Acumulados'] = serie_tiempo['Casos Diarios'].cumsum()

plt.figure(figsize=(14, 7))
plt.plot(serie_tiempo.index, serie_tiempo['Casos Reales Acumulados'], color='blue', linewidth=2, label='Casos Reales Acumulados')
plt.plot(serie_tiempo.index, serie_tiempo['Casos Predichos Acumulados'], color='red', linewidth=2, label='Casos Predichos Acumulados')
plt.title(f"Casos Acumulados Reales vs Predicciones en {pais}", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Casos Acumulados", fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
plt.show()

# **Exportar resultados**
# Propósito: Guardar las predicciones y los casos reales en un archivo CSV para análisis posterior.
ruta_salida = r'C:\Users\ben19\Downloads\codigoIA\resultados_regresion_lineal.csv'
serie_tiempo.to_csv(ruta_salida, index=True)
print(f"Resultados exportados a {ruta_salida}")
