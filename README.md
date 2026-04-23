## INFORME DE LABORATORIO #4.
# SEÑALES ELECTROMIOGRAFICAS EMG

### DESCRIPCIÓN 
La presente práctica de laboratorio tiene como objetivo el análisis de señales electromiográficas (EMG) reales con el fin de comprender el comportamiento de la actividad muscular tanto en el dominio del tiempo como en el dominio de la frecuencia. A partir de una señal adquirida experimentalmente mediante sensores adecuados, se realiza un proceso de visualización, tratamiento y análisis que permite extraer información relevante sobre la dinámica muscular.

En una primera etapa, la señal es observada en el dominio temporal para identificar sus características principales, tales como variaciones de amplitud, presencia de ruido y patrones asociados a la contracción muscular. Posteriormente, se lleva a cabo un preprocesamiento básico que incluye la eliminación de valores atípicos y el ajuste del nivel de referencia de la señal, con el fin de garantizar condiciones adecuadas para su análisis.

En etapas posteriores, la señal es segmentada en diferentes intervalos representativos de la actividad muscular. Sobre estos segmentos se aplican herramientas de análisis en el dominio de la frecuencia, particularmente la Transformada Rápida de Fourier (FFT), con el objetivo de obtener el espectro de amplitud y estudiar la distribución de energía en función de la frecuencia.

A partir de este análisis espectral, se comparan distintos segmentos de la señal para identificar cambios asociados a fenómenos fisiológicos como la fatiga muscular. Se evalúan parámetros como la energía en altas frecuencias y el comportamiento del pico espectral, los cuales permiten inferir modificaciones en la actividad de las fibras musculares durante esfuerzos sostenidos.

Finalmente, se discute la relevancia del análisis de señales EMG como herramienta en el campo de la ingeniería biomédica, destacando su utilidad para el estudio no invasivo de la función muscular y su aplicación en contextos clínicos, deportivos y de investigación.


### OBJETIVOS
Identificar cambios en las características espectrales de una señal electromiográfica (EMG) cuando se alcanza la fatiga muscular.
Aplicar el filtrado de señales continuas para el procesamiento una señal electromiográfica (EMG). 
Detectar la aparición de fatiga muscular mediante el análisis espectral de contracciones musculares individuales.
Comparar el comportamiento de una señal emulada y una señal real en términos de frecuencia media y mediana.
Emplear herramientas computacionales para el procesamiento, segmentación y análisis de señales biomédicas. 


###  PARTE C.
En la Parte C de la práctica se realiza el análisis espectral de la señal electromiográfica (EMG) mediante la aplicación de la Transformada Rápida de Fourier (FFT), con el objetivo de estudiar el contenido frecuencial asociado a la actividad muscular.

Para ello, la señal EMG previamente adquirida es segmentada en diferentes intervalos que representan distintas etapas de la contracción muscular. A cada uno de estos segmentos se le aplica la FFT, obteniendo así su espectro de amplitud, el cual permite visualizar la distribución de la energía en función de la frecuencia.

A partir de los espectros obtenidos, se lleva a cabo una comparación entre los primeros y últimos segmentos de la señal con el fin de identificar cambios en el contenido frecuencial asociados a la fatiga muscular. En particular, se analiza la variación en la energía de altas frecuencias, la cual tiende a disminuir a medida que el músculo se fatiga.

Adicionalmente, se evalúa el comportamiento del pico espectral para determinar posibles desplazamientos en la frecuencia dominante de la señal. Este análisis permite inferir cambios en la actividad fisiológica del músculo durante un esfuerzo sostenido.

Finalmente, los resultados obtenidos son interpretados para destacar la utilidad del análisis espectral como herramienta en la electromiografía, permitiendo identificar características relevantes de la función muscular a partir del procesamiento de señales.


# DIAGRAMA
<img width="300" height="400" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/Diagrama1.png" />
<img width="300" height="400" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/Diagrama2.png" />

# CÓDIGO
## Conectar Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```
En esta sección se establece la conexión entre el entorno de Google Colab y Google Drive, permitiendo acceder a los archivos almacenados en la nube. Esto es necesario para cargar la señal EMG previamente adquirida y almacenada en formato CSV.
## Importar librerías
```
# =========================================
# 2. LIBRERÍAS
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
Se importan las librerías necesarias para el procesamiento de la señal. La librería NumPy se utiliza para realizar operaciones numéricas y manejo de arreglos, Pandas permite la lectura y manipulación de archivos de datos, y Matplotlib se emplea para la visualización gráfica de la señal y sus transformaciones.
## Cargar datos
```
# =========================================
# 3. CARGAR DATOS
# =========================================
ruta = "/content/drive/MyDrive/valentina emg real_20260327_141412 (1).csv"
data = pd.read_csv(ruta)

t = data.iloc[:,0].values
emg = data.iloc[:,1].values
```
En esta etapa se carga el archivo que contiene la señal EMG. Se extraen dos columnas principales: el tiempo, que representa el eje temporal de la señal, y el voltaje, que corresponde a la señal electromiográfica medida. Estos datos son convertidos a arreglos numéricos para su posterior procesamiento.
## Limpieza de la señal 
```
# =========================================
# 4. LIMPIEZA
# =========================================
emg = np.nan_to_num(emg)
emg = emg - np.mean(emg)

fs = 1 / (t[1] - t[0])
```
Se realiza un preprocesamiento básico de la señal. En primer lugar, se eliminan valores no válidos como NaN. Posteriormente, se remueve el componente de corriente continua (offset) restando el valor medio de la señal, lo que permite centrarla alrededor de cero. Finalmente, se calcula la frecuencia de muestreo a partir del intervalo entre muestras consecutivas.
## Visualización en el dominio del tiempo
```
# =========================================
# 5. GRAFICAR SEÑAL
# =========================================
plt.figure(figsize=(10,4))
plt.plot(t, emg, color="#E75480")
plt.title("Señal EMG real (sin filtrar)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid()
plt.show()

```
Se grafica la señal EMG en función del tiempo con el objetivo de observar su comportamiento en el dominio temporal. Esta visualización permite identificar características como la variabilidad de la señal, la presencia de ruido y posibles patrones asociados a la actividad muscular.

## Segmentacion de la señal
```
# =========================================
# 6. SEGMENTACIÓN POR BLOQUES 
# =========================================
n_segmentos = 5
segmentos = np.array_split(emg, n_segmentos)
```
La señal se divide en varios segmentos de igual longitud con el fin de analizar diferentes intervalos de tiempo de forma independiente. Esta segmentación permite comparar el comportamiento de la señal en distintas etapas de la actividad muscular..
## FFT
```
# =========================================
# 7. FFT
# =========================================
def calcular_fft(signal, fs):
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals[:N//2])
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    return freqs, fft_vals

color_rosa = "#E75480"
```
Se define una función para calcular la Transformada Rápida de Fourier (FFT) de cada segmento de la señal. Este proceso permite transformar la señal del dominio del tiempo al dominio de la frecuencia, obteniendo así las componentes frecuenciales presentes y su respectiva magnitud.
## Espectro de cada segmento
```# =========================================
# 8. FFT DE CADA SEGMENTO
# =========================================
for i, seg in enumerate(segmentos):
    freqs, fft_vals = calcular_fft(seg, fs)

    plt.figure(figsize=(8,4))
    plt.semilogy(freqs, fft_vals, color=color_rosa)
    plt.title(f"FFT - Segmento {i+1}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud (log)")
    plt.grid()
    plt.show()
```
Se calcula y grafica el espectro de amplitud para cada segmento de la señal utilizando una escala logarítmica. Esta representación facilita la visualización de componentes de baja y alta magnitud, permitiendo analizar la distribución de energía en diferentes frecuencias.

## Comparacion espectral
```
# =========================================
# 9. COMPARACIÓN
# =========================================
f1, fft1 = calcular_fft(segmentos[0], fs)
f2, fft2 = calcular_fft(segmentos[-1], fs)

plt.figure(figsize=(8,4))
plt.semilogy(f1, fft1, label="Inicio", color=color_rosa)
plt.semilogy(f2, fft2, '--', label="Final", color="#8B3A62")

plt.title("Comparación espectral")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud (log)")
plt.legend()
plt.grid()
plt.show()
```
Se comparan los espectros correspondientes al primer y último segmento de la señal. Esta comparación permite identificar cambios en el contenido frecuencial a lo largo del tiempo, lo cual puede estar asociado a fenómenos como la fatiga muscular.

## Energia en altas frecuencias 
```
# =========================================
# 10. ENERGÍA ALTA FRECUENCIA
# =========================================
def energia_alta(freqs, fft_vals):
    return np.sum(fft_vals[freqs > 100])

print("\n--- Energía alta frecuencia ---")
for i, seg in enumerate(segmentos):
    f, fft_vals = calcular_fft(seg, fs)
    print(f"Segmento {i+1}: {energia_alta(f, fft_vals):.4f}")
```
Se calcula la energía de la señal en el rango de altas frecuencias (mayores a 100 Hz). Este parámetro es relevante en el análisis de fatiga muscular, ya que una disminución en este contenido frecuencial suele estar asociada a la reducción de la actividad de las fibras musculares de alta velocidad.
## Pico Espectral
```
# =========================================
# 11. PICO ESPECTRAL
# =========================================
picos = []

print("\n--- Pico espectral ---")
for i, seg in enumerate(segmentos):
    f, fft_vals = calcular_fft(seg, fs)
    pico = f[np.argmax(fft_vals)]
    picos.append(pico)
    print(f"Segmento {i+1}: {pico:.2f} Hz")
```
Se determina la frecuencia correspondiente al valor máximo del espectro de amplitud, conocida como pico espectral. Este valor indica la frecuencia dominante en cada segmento de la señal y permite analizar posibles cambios en la actividad muscular.
## Desplazamiento del pico
```
# =========================================
# 12. DESPLAZAMIENTO
# =========================================
plt.figure(figsize=(8,4))
plt.plot(range(1, len(picos)+1), picos, 'o-', color=color_rosa)

plt.title("Desplazamiento del pico espectral")
plt.xlabel("Segmento")
plt.ylabel("Frecuencia [Hz]")
plt.grid()
plt.show()
```
Se grafica la evolución del pico espectral a lo largo de los diferentes segmentos. Este análisis permite observar posibles desplazamientos en la frecuencia dominante de la señal, los cuales pueden estar relacionados con el esfuerzo sostenido y la fatiga muscular.
# GRAFICAS

<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen2.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen3.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen4.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen5.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen6.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen7.png" />
<img width="500" height="200" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen8.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen9.png" />


### PARTE B

En esta fase se realizó la adquisición de la señal electromiográfica (EMG) proveniente de un sujeto voluntario sano, colocando los electrodos sobre un grupo muscular específico (como el bíceps o antebrazo). Durante el registro, el sujeto efectuó contracciones repetidas hasta la aparición de la fatiga muscular, permitiendo analizar cómo varían las componentes frecuenciales de la señal en condiciones reales

#### SEÑAL CAPTURADA

![Señal capturada](https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen_2026-04-12_205056906.png?raw=true)

Luego de la captura de la señal se filtro, se realizaron los calculos del orden manualmente para luego programar el filtro.

![FILTRO](https://github.com/estmanuelamancera/Lab4-2026/blob/main/WhatsApp%20Image%202026-04-22%20at%208.47.53%20PM.jpeg?raw=true)

#### PROGRAMACIÓN

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

try:
    df = pd.read_csv('VALEMG.csv')
    emg_raw = df.iloc[:, 1].values  # Cambia el índice si tu EMG está en otra columna
    # fs = 1 / (df.iloc[1, 0] - df.iloc[0, 0]) # Intenta calcular fs automáticamente
    fs = 2500 
except:
    print("Error al cargar. Asegúrate de subir el archivo al panel izquierdo de Colab.")

# 2. Diseño del Filtro (especificaciones)
low_cut = 20    # Hz
high_cut = 450  # Hz
nyquist = 0.5 * fs
low = low_cut / nyquist
high = high_cut / nyquist

# Creamos el filtro de orden 2 (calculado)
b, a = butter(2, [low, high], btype='band')

# Aplicamos el filtro (filtfilt evita el desfase temporal)
emg_filtered = filtfilt(b, a, emg_raw)

# 3. Visualización
plt.figure(figsize=(15, 8))

# Subplot 1: Señal Original
plt.subplot(2, 1, 1)
plt.plot(emg_raw, color='gray', alpha=0.5, label='EMG Raw (Cruda)')
plt.title('Señal EMG Original (Antes del Filtro)')
plt.ylabel('Voltaje (mV)')
plt.legend()
plt.grid(True)

# Subplot 2: Señal Filtrada
plt.subplot(2, 1, 2)
plt.plot(emg_filtered, color='blue', label='EMG Filtrada (20-450 Hz)')
plt.title('Señal EMG después del Filtro Butterworth (Orden 2)')
plt.xlabel('Muestras')
plt.ylabel('Voltaje (mV)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```

#### FILTRO SEÑAL

![FILTRO SEÑAL](https://github.com/estmanuelamancera/Lab4-2026/blob/main/image.png?raw=true)






## CONCLUSIONES


El análisis espectral de la señal electromiográfica mediante la Transformada Rápida de Fourier (FFT) permite identificar la distribución de energía en función de la frecuencia, constituyéndose como una herramienta útil para el estudio de la actividad muscular. A través de este enfoque, es posible analizar cambios en el contenido frecuencial de la señal asociados a fenómenos fisiológicos como la fatiga muscular.

En los resultados obtenidos, se evidenció una tendencia general a la disminución de la energía en altas frecuencias en los últimos segmentos de la señal, lo cual es consistente con el comportamiento esperado durante un esfuerzo sostenido, donde se presenta una reducción en la velocidad de conducción de las fibras musculares y en la participación de fibras de contracción rápida.

Sin embargo, el análisis del pico espectral mostró una frecuencia dominante constante alrededor de 60 Hz, la cual no corresponde a la actividad muscular, sino a la interferencia de la red eléctrica. Este resultado pone en evidencia la importancia del preprocesamiento de la señal, especialmente la aplicación de filtros adecuados, para eliminar componentes no fisiológicos que pueden afectar la interpretación del espectro.

En conclusión, el análisis espectral mediante FFT es una herramienta valiosa en electromiografía, ya que permite extraer información relevante sobre el estado funcional del músculo. No obstante, su correcta aplicación requiere un adecuado tratamiento de la señal para garantizar que los resultados reflejen fielmente la actividad fisiológica y no estén influenciados por fuentes externas de ruido.

