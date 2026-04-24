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


### PARTE A

En esta primera etapa, se configuró el generador de señales biológicas en modo electromiografía (EMG) con el objetivo de simular cinco contracciones musculares voluntarias. Este procedimiento permite reproducir de manera controlada la actividad eléctrica generada por el músculo durante contracciones sucesivas. Previo a esto se segmento la señal, se calculo la frecuencia media y mediana para cada contraccion para luego graficar y tabular los resultados obtenidos.

### SEÑAL CAPTURADA 

![]()

### SEGMENTACIÓN DE LA SEÑAL 

![SEGMENTACIÓN]()


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
### Fatiga muscular 
La fatiga muscular, en el contexto del análisis espectral de señales EMG, se asocia típicamente con un desplazamiento del contenido frecuencial hacia bajas frecuencias, evidenciado por una disminución progresiva de la frecuencia media, la frecuencia mediana y el pico espectral a lo largo del tiempo. Este fenómeno está relacionado con la reducción en la velocidad de conducción de las fibras musculares y cambios en la sincronización de las unidades motoras durante un esfuerzo sostenido. Sin embargo, a partir de los resultados obtenidos en este análisis, no se observa una tendencia clara y sostenida de disminución en las frecuencias características. Por el contrario, los valores de frecuencia media, mediana y pico espectral presentan variaciones a lo largo de las contracciones, incluyendo incrementos en ciertos segmentos intermedios y finales.

Este comportamiento sugiere que la señal analizada no corresponde a un esfuerzo muscular completamente constante o controlado, lo cual puede dificultar la identificación de patrones clásicos de fatiga. Factores como variaciones en la intensidad de la contracción, pausas involuntarias o cambios en el reclutamiento de unidades motoras pueden influir en la distribución espectral de la señal. En consecuencia, aunque la metodología empleada es adecuada para la detección de fatiga muscular, en este caso particular no se evidencia de forma concluyente un proceso de fatiga, sino más bien una variabilidad en la activación neuromuscular durante el registro.
### Cálculo del Desplazamiento del Pico Espectral
El desplazamiento del pico espectral hace referencia al cambio en la frecuencia donde se concentra la mayor energía del espectro de la señal. En señales EMG, este parámetro es comúnmente utilizado como indicador de fatiga muscular, ya que bajo condiciones de esfuerzo sostenido suele presentarse un desplazamiento hacia frecuencias más bajas debido a la reducción de las componentes de alta frecuencia.

En el aplicativo desarrollado en Python, se implementó el siguiente código para calcular el desplazamiento del pico espectral:
```python
if len(picos) >= 2:
    desplazamiento = picos[-1] - picos[0]
    print(f"Desplazamiento del pico espectral: {desplazamiento:.2f} Hz")
else:
    print("No hay suficientes contracciones para calcular el desplazamiento.")
```
<img width="1348" height="1682" alt="image" src="https://github.com/user-attachments/assets/d87c8ee2-a364-4eb4-a18f-d16ef58f3dd8" />
A diferencia del comportamiento esperado en presencia de fatiga muscular, donde el pico espectral tiende a desplazarse hacia frecuencias más bajas, el resultado obtenido no evidencia una disminución progresiva de la frecuencia dominante. Aunque se observa un cambio en la frecuencia pico de 11.05 Hz, este no sigue una tendencia claramente decreciente, sino que forma parte de la variabilidad observada a lo largo de las contracciones. Este comportamiento puede estar asociado a cambios en la intensidad del esfuerzo, variaciones en el reclutamiento de unidades motoras o a la falta de un esfuerzo muscular constante durante la adquisición de la señal.

En consecuencia, aunque el desplazamiento del pico espectral es una herramienta útil para el análisis de fatiga muscular, en este caso no permite concluir de manera contundente la presencia de fatiga, sino que refleja principalmente la variabilidad en la activación muscular.


## CONCLUSIONES


El análisis espectral de la señal electromiográfica mediante la Transformada Rápida de Fourier (FFT) permite identificar la distribución de energía en función de la frecuencia, constituyéndose como una herramienta útil para el estudio de la actividad muscular. A través de este enfoque, es posible analizar cambios en el contenido frecuencial de la señal asociados a fenómenos fisiológicos como la fatiga muscular.

En los resultados obtenidos, se evidenció una tendencia general a la disminución de la energía en altas frecuencias en los últimos segmentos de la señal, lo cual es consistente con el comportamiento esperado durante un esfuerzo sostenido, donde se presenta una reducción en la velocidad de conducción de las fibras musculares y en la participación de fibras de contracción rápida.

Sin embargo, el análisis del pico espectral mostró una frecuencia dominante constante alrededor de 60 Hz, la cual no corresponde a la actividad muscular, sino a la interferencia de la red eléctrica. Este resultado pone en evidencia la importancia del preprocesamiento de la señal, especialmente la aplicación de filtros adecuados, para eliminar componentes no fisiológicos que pueden afectar la interpretación del espectro.

En conclusión, el análisis espectral mediante FFT es una herramienta valiosa en electromiografía, ya que permite extraer información relevante sobre el estado funcional del músculo. No obstante, su correcta aplicación requiere un adecuado tratamiento de la señal para garantizar que los resultados reflejen fielmente la actividad fisiológica y no estén influenciados por fuentes externas de ruido.
El análisis espectral de la señal electromiográfica (EMG) mediante la Transformada Rápida de Fourier (FFT) permitió caracterizar la distribución de energía en el dominio de la frecuencia para cada contracción muscular. Se observó que la mayor concentración de energía se encuentra en el rango de 20 a 100 Hz, lo cual es consistente con el comportamiento fisiológico típico de este tipo de señales.

La segmentación de la señal a través del cálculo de la envolvente RMS permitió identificar de manera adecuada las contracciones musculares, facilitando el análisis individual de cada evento y la comparación entre diferentes momentos del esfuerzo.

A partir del cálculo de las frecuencias características (frecuencia media, mediana y pico espectral), se evidenció una variabilidad en el contenido frecuencial a lo largo de las contracciones. Sin embargo, no se observó una disminución progresiva y sostenida de estos parámetros, lo cual indica que no se presenta un patrón claro de fatiga muscular en la señal analizada.

El análisis del desplazamiento del pico espectral mostró un cambio total de 11.05 Hz; no obstante, este desplazamiento no siguió una tendencia decreciente, sino que reflejó variaciones en la activación muscular. Esto sugiere que el esfuerzo no fue completamente constante o que existieron cambios en el reclutamiento de unidades motoras durante el registro.

En conclusión, el análisis espectral constituye una herramienta útil para evaluar la dinámica de la actividad muscular y detectar posibles cambios asociados a la fatiga. Sin embargo, su efectividad depende de la estabilidad del esfuerzo realizado, por lo que se recomienda, para futuros estudios, trabajar bajo condiciones experimentales más controladas que permitan evidenciar de manera más clara los efectos de la fatiga muscular.

### PREGUNTAS PARA LA DISCUSIÓN 

¿Cambian los valores de frecuencia media y mediana a medida que el músculo se acerca a la fatiga? ¿A qué podría atribuirse este cambio? 

En condiciones ideales, cuando el músculo se aproxima a la fatiga, se espera una disminución progresiva de la frecuencia media y la frecuencia mediana de la señal EMG. Este comportamiento se debe principalmente a la reducción en la velocidad de conducción de las fibras musculares y a cambios en la sincronización y reclutamiento de las unidades motoras, lo que provoca una mayor concentración de energía en bajas frecuencias.

¿Cómo justifica el uso de herramientas como la transformada de Fourier en escenarios como, por ejemplo, terapias de rehabilitación? 

El uso de herramientas como la Transformada de Fourier en el análisis de señales EMG es fundamental en escenarios de rehabilitación, ya que permite estudiar la actividad muscular en el dominio de la frecuencia, proporcionando información que no es evidente en el dominio del tiempo. En terapias de rehabilitación, esta información puede ser utilizada para ajustar la intensidad de los ejercicios, prevenir la fatiga excesiva y optimizar los programas de entrenamiento. Además, permite realizar un seguimiento objetivo del progreso del paciente, facilitando la toma de decisiones clínicas basadas en datos cuantitativos.
En este sentido, la transformada de Fourier se convierte en una herramienta clave para el análisis y la evaluación funcional del sistema neuromuscular en aplicaciones biomédicas.

