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

try: I
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

### SEGMENTACION Y FRECUENCIAS
La señal de electromiografía fue segmentada en un total de 35 contracciones musculares. Debido a la naturaleza de la muestra, se optó por analizar tres momentos clave que representan la evolución temporal del ejercicio: la contracción inicial, la intermedia y la final. Para cada uno de estos segmentos, se realizó una caracterización espectral mediante el cálculo de la frecuencia media y la frecuencia mediana, permitiendo así identificar posibles desplazamientos en el espectro asociados a la fatiga muscular.

### PROGRAMACIÓN

```python

import matplotlib.pyplot as plt
import numpy as np

# 1. Asegurar la detección de los picos (Ajuste de sensibilidad)
# Hacemos esto para que el detector encuentre las 35 contracciones
emg_abs = np.abs(emg_filtered)
ventana_suave = int(0.3 * 2500) 
envolvente = np.convolve(emg_abs, np.ones(ventana_suave)/ventana_suave, mode='same')

from scipy.signal import find_peaks
# Bajamos la altura para asegurarnos de captar las 35
indices_picos, _ = find_peaks(envolvente, distance=2500*0.8, height=np.mean(envolvente)*0.4)

# 2. Definir los objetivos (Contracción 1, 11 y 34)
# Usamos try/except para que si solo encuentra 4, te grafique las que pueda sin romperse
puntos_interes = [0, 10, 33] 
nombres = ['Contracción 1', 'Contracción 11', 'Contracción 34']
colores = ['#1f77b4', '#ff7f0e', '#d62728']
fs = 2500

# 3. Graficar los segmentos en el dominio del tiempo
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

for i, idx_target in enumerate(puntos_interes):
    if idx_target < len(indices_picos):
        p = indices_picos[idx_target]
        # Tomamos un rango de 1 segundo alrededor del pico (0.5s antes y 0.5s después)
        inicio, fin = max(0, p - 1250), min(len(emg_filtered), p + 1250)
        segmento = emg_filtered[inicio:fin]
        
        # Crear eje de tiempo para ese segmento específico
        t_seg = np.arange(len(segmento)) / fs
        
        axes[i].plot(t_seg, segmento, color=colores[i], lw=1)
        axes[i].set_title(f"{nombres[i]} (Centro en t = {p/fs:.2f} s)", fontweight='bold')
        axes[i].set_ylabel('Amplitud (mV)')
        axes[i].grid(True, alpha=0.3)
    else:
        axes[i].text(0.5, 0.5, f'Contracción {idx_target+1} no detectada\n(Ajusta la sensibilidad)', 
                     ha='center', va='center', fontsize=12, color='red')

axes[2].set_xlabel('Tiempo del segmento (s)')
plt.tight_layout()
plt.show()

```
### PARTES DE LA SEÑAL 
![Segmentacion](https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen_2026-04-22_230622625.png?raw=true)

### MEDIA Y MEDIANA 

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# 1. Usamos los mismos índices que definimos para los subplots
indices_a_graficar = [0, total_picos // 2, total_picos - 1]
etiquetas = ['Inicio (C1)', 'Mitad (C18)', 'Final (C35)']
colores_puntos = ['#1f77b4', '#ff7f0e', '#d62728']

f_meds_resumen = []
f_medias_resumen = []

# 2. Calculamos los datos solo para esos 3 momentos
for idx in indices_a_graficar:
    p = indices_picos[idx]
    inicio, fin = max(0, p - 1000), min(len(emg_filtered), p + 1000)
    segmento = emg_filtered[inicio:fin]
    
    f_w, psd = welch(segmento, fs=2500, nperseg=1024)
    
    # Frecuencia Mediana
    suma_acum = np.cumsum(psd)
    f_mediana = f_w[np.where(suma_acum >= suma_acum[-1] / 2)[0][0]]
    
    # Frecuencia Media
    f_media = np.sum(f_w * psd) / np.sum(psd)
    
    f_meds_resumen.append(f_mediana)
    f_medias_resumen.append(f_media)

# 3. Graficar el resumen de los 3 momentos
plt.figure(figsize=(8, 5))

# Graficamos líneas que unan solo los 3 puntos
plt.plot(etiquetas, f_meds_resumen, 'o-', label='Frecuencia Mediana', color='blue', markersize=10)
plt.plot(etiquetas, f_medias_resumen, 's-', label='Frecuencia Media', color='red', markersize=10)

# Añadir etiquetas de valor sobre los puntos para mayor claridad
for i, val in enumerate(f_meds_resumen):
    plt.text(i, val + 1, f'{val:.1f} Hz', ha='center', color='blue', fontweight='bold')

plt.title('Tendencia de Frecuencias en Momentos Clave', fontweight='bold')
plt.ylabel('Frecuencia (Hz)')
plt.ylim(min(f_meds_resumen + f_medias_resumen) - 10, max(f_meds_resumen + f_medias_resumen) + 10)
plt.grid(True, alpha=0.2)
plt.legend()
plt.show()

```
### RESULTADO
![RESULTADO MEDIA Y MEDIANA](https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen_2026-04-22_231023635.png?raw=true)

### FRECUENCIAS

```python

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# 1. Selección dinámica a prueba de errores
total_picos = len(indices_picos)

if total_picos < 3:
    print(f"¡OJO! Solo se detectaron {total_picos} contracciones en la señal.")
    # Selecciona las pocas que haya encontrado
    indices_a_graficar = list(range(total_picos))
else:
    # Selecciona dinámicamente la inicial, la del medio y la final de las que existan
    idx_medio = total_picos // 2
    idx_final = total_picos - 1
    indices_a_graficar = [0, idx_medio, idx_final]

nombres = ['Contracción Inicial', 'Contracción Media', 'Contracción Final']
colores = ['#1f77b4', '#ff7f0e', '#d62728']
fs = 2500  

# 2. Crear SIEMPRE la figura asegurando la estructura 2D (Filas x 2 Columnas)
filas = max(len(indices_a_graficar), 1)
fig, axes = plt.subplots(filas, 2, figsize=(15, 4 * filas), sharex='col')

# Forzar a que axes sea siempre una matriz bidimensional (incluso si solo hay 1 fila)
if filas == 1:
    axes = np.array([axes])

for i, idx_c in enumerate(indices_a_graficar):
    p = indices_picos[idx_c]
    tiempo_seg = p / fs
    
    # Extraer segmento
    inicio, fin = max(0, p - 1000), min(len(emg_filtered), p + 1000)
    segmento = emg_filtered[inicio:fin]

    # ==========================================
    # COLUMNA 1 (IZQUIERDA): PSD WELCH
    # ==========================================
    f_w, psd = welch(segmento, fs=fs, nperseg=1024)
    psd_norm = psd / np.max(psd)

    axes[i, 0].plot(f_w, psd_norm, color=colores[i % 3], lw=2)
    axes[i, 0].fill_between(f_w, psd_norm, color=colores[i % 3], alpha=0.2)
    
    # Calcular Frecuencia Mediana de forma automática y segura
    suma_acumulada = np.cumsum(psd)
    idx_mediana = np.where(suma_acumulada >= suma_acumulada[-1] / 2)[0][0]
    f_med = f_w[idx_mediana]
    
    axes[i, 0].axvline(x=f_med, color='black', linestyle='--', alpha=0.8, label=f'F. Med: {f_med:.1f} Hz')
    axes[i, 0].legend(loc='upper right')
    
    # Titulo dinámico según la cantidad de contracciones reales
    titulo_izq = nombres[i] if filas == 3 else f"Contracción {idx_c + 1}"
    axes[i, 0].set_title(f"{titulo_izq} | PSD Welch", fontweight='bold')
    axes[i, 0].set_ylabel('Potencia Norm.')
    axes[i, 0].set_xlim(0, 300)
    axes[i, 0].grid(True, alpha=0.3)

    # ==========================================
    # COLUMNA 2 (DERECHA): FFT
    # ==========================================
    N = len(segmento)
    yf = np.fft.fft(segmento)
    xf = np.fft.fftfreq(N, 1/fs)

    xf_pos = xf[:N//2]
    yf_abs = (2.0/N) * np.abs(yf[:N//2])
    yf_norm = yf_abs / np.max(yf_abs) 

    axes[i, 1].plot(xf_pos, yf_norm, color=colores[i % 3], lw=1.5)
    axes[i, 1].set_title(f"Espectro FFT | Ocurrió en: {tiempo_seg:.1f} s", fontweight='bold')
    axes[i, 1].set_ylabel('Amplitud Norm.')
    axes[i, 1].set_xlim(0, 300)
    axes[i, 1].grid(True, alpha=0.3)

# Configuración de los ejes X solo en la parte inferior
axes[-1, 0].set_xlabel('Frecuencia (Hz) <- Distribución de energía')
axes[-1, 1].set_xlabel('Frecuencia (Hz) <- Componentes FFT')

plt.tight_layout()
plt.show()

```
### VISUALIZACIÓN FRECUENCIA 

![FRECUENCIAS](https://github.com/estmanuelamancera/Lab4-2026/blob/main/imagen_2026-04-22_231406757.png?raw=true)

### DISCUSIÓN

Disminución de la Velocidad de Conducción: La acumulación de metabolitos (ácido láctico e iones de hidrógeno) genera una acidosis metabólica que reduce el pH del músculo. Esto hace que el potencial de acción se desplace más lento por la membrana muscular, "comprimiendo" el espectro hacia las frecuencias bajas (desplazamiento a la izquierda en la FFT).

Falla de Fibras Rápidas (Tipo II): Al inicio y mitad del ejercicio (61.0 Hz), el cuerpo recluta fibras rápidas para mantener la fuerza. Sin embargo, estas son las primeras en agotarse. Al final del protocolo (C35), predominan las fibras lentas (Tipo I), que operan naturalmente a frecuencias menores.

Sincronización de Unidades Motoras: Como mecanismo de defensa, el sistema nervioso sincroniza los disparos de las fibras restantes para intentar sostener la tensión. Aunque esto mantiene la fuerza momentáneamente, reduce la variabilidad de la señal y concentra la energía en las bandas de frecuencia más bajas.
## PARTE C
En esta sección se realizó el análisis en el dominio de la frecuencia de la señal electromiográfica (EMG) mediante la aplicación de la Transformada Rápida de Fourier (FFT) a cada una de las contracciones musculares previamente detectadas. Para ello, la señal fue preprocesada mediante filtrado notch y pasabanda con el fin de eliminar ruido e interferencias, y posteriormente se segmentó utilizando la envolvente RMS para aislar los periodos de actividad muscular.

A partir de estos segmentos, se calculó el espectro de amplitud (frecuencia vs. magnitud), lo que permitió caracterizar la distribución de energía en el dominio frecuencial. Este análisis se utilizó para comparar las primeras y últimas contracciones, evaluar posibles cambios asociados al esfuerzo sostenido y estudiar el comportamiento del pico espectral como indicador de la dinámica muscular.
### Transformada Rápida de Fourier (FFT) a cada contracción 
Para analizar el contenido frecuencial de la señal electromiográfica (EMG), se aplicó la Transformada Rápida de Fourier (FFT) a cada una de las contracciones musculares previamente segmentadas. Antes de realizar la transformada, cada segmento fue centrado eliminando su componente de corriente directa (DC) y se le aplicó una ventana de Hamming con el fin de reducir el efecto de fuga espectral.

Posteriormente, se calculó la FFT de cada contracción, obteniendo así su representación en el dominio de la frecuencia. A partir de este resultado, se extrajo el espectro de magnitud considerando únicamente las frecuencias positivas, lo que permitió analizar la distribución de energía de la señal EMG en función de la frecuencia.
Este procedimiento se repitió para todas las contracciones detectadas, permitiendo un análisis individual de cada evento muscular y facilitando la comparación entre diferentes momentos del esfuerzo.
```python
def calcular_fft(signal, fs):
    N = len(signal)

    # Eliminar componente DC
    signal = signal - np.mean(signal)

    # Aplicar ventana de Hamming
    ventana = np.hamming(N)
    signal = signal * ventana

    # FFT
    X = np.fft.fft(signal)
    f = np.fft.fftfreq(N, d=1/fs)

    # Frecuencias positivas
    mask = f >= 0
    f = f[mask]
    X = X[mask]

    # Magnitud normalizada
    mag = (2 / N) * np.abs(X)

    return f, mag
```
La función calcular_fft permite obtener el espectro de amplitud de cada contracción de la señal EMG en el dominio de la frecuencia. Inicialmente, se elimina la componente DC restando el valor promedio de la señal. Luego, se aplica una ventana de Hamming para reducir la fuga espectral. Posteriormente, se calcula la Transformada Rápida de Fourier (FFT), obteniendo la representación de la señal en frecuencia. Se seleccionan únicamente las frecuencias positivas y se calcula la magnitud normalizada del espectro.
Finalmente, la función retorna el vector de frecuencias y su correspondiente magnitud, permitiendo analizar la distribución de energía de la señal EMG.
### Gráficas del Espectro de Amplitud
A continuación se presentarán las tres primeras y últimas gráficas del espectro de amplitud en función de la frecuencia, para realizar su respectivo análisis y comparación.
<img width="1343" height="1634" alt="image" src="https://github.com/user-attachments/assets/8ad25bbf-a159-4e57-a555-f863e71d9168" />
#### Analisis
A partir del cálculo de las frecuencias características (frecuencia media, frecuencia mediana y pico espectral) para cada contracción, se observa que la señal EMG presenta variabilidad en su contenido frecuencial a lo largo del tiempo. En las primeras contracciones (1–3), la frecuencia media se mantiene alrededor de 40–47 Hz, la frecuencia mediana entre 30 y 35 Hz y el pico espectral entre 22 y 27 Hz. Esto indica que la energía de la señal está concentrada principalmente en bajas y medias frecuencias durante el inicio del esfuerzo muscular.

En las contracciones intermedias (4–10), se evidencia una mayor dispersión en los valores, destacándose incrementos en la frecuencia media (hasta aproximadamente 60–63 Hz) y en la frecuencia mediana (superando los 50 Hz en algunos casos), así como picos espectrales más altos, alcanzando valores cercanos a 60 Hz. Este comportamiento sugiere una mayor variabilidad en la activación muscular, posiblemente asociada a cambios en la intensidad de la contracción o en el reclutamiento de unidades motoras.En las últimas contracciones (14–16), las frecuencias características tienden a estabilizarse nuevamente en rangos cercanos a los iniciales, con frecuencias medias entre 42 y 54 Hz, medianas entre 34 y 44 Hz y picos espectrales entre 31 y 40 Hz.

Al analizar la evolución global de los parámetros, no se observa una disminución progresiva y sostenida de las frecuencias características, lo cual sería indicativo de fatiga muscular. En cambio, se evidencia un comportamiento variable, con aumentos y disminuciones a lo largo del tiempo, lo que sugiere que el esfuerzo muscular no fue completamente constante o que existieron cambios en la dinámica de activación neuromuscular durante el registro.

<img width="1348" height="1682" alt="image" src="https://github.com/user-attachments/assets/d87c8ee2-a364-4eb4-a18f-d16ef58f3dd8" />



## CONCLUSIONES


El análisis espectral de la señal electromiográfica mediante la Transformada Rápida de Fourier (FFT) permite identificar la distribución de energía en función de la frecuencia, constituyéndose como una herramienta útil para el estudio de la actividad muscular. A través de este enfoque, es posible analizar cambios en el contenido frecuencial de la señal asociados a fenómenos fisiológicos como la fatiga muscular.

En los resultados obtenidos, se evidenció una tendencia general a la disminución de la energía en altas frecuencias en los últimos segmentos de la señal, lo cual es consistente con el comportamiento esperado durante un esfuerzo sostenido, donde se presenta una reducción en la velocidad de conducción de las fibras musculares y en la participación de fibras de contracción rápida.

Sin embargo, el análisis del pico espectral mostró una frecuencia dominante constante alrededor de 60 Hz, la cual no corresponde a la actividad muscular, sino a la interferencia de la red eléctrica. Este resultado pone en evidencia la importancia del preprocesamiento de la señal, especialmente la aplicación de filtros adecuados, para eliminar componentes no fisiológicos que pueden afectar la interpretación del espectro.

En conclusión, el análisis espectral mediante FFT es una herramienta valiosa en electromiografía, ya que permite extraer información relevante sobre el estado funcional del músculo. No obstante, su correcta aplicación requiere un adecuado tratamiento de la señal para garantizar que los resultados reflejen fielmente la actividad fisiológica y no estén influenciados por fuentes externas de ruido.

