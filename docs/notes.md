# Notas del proyecto

Las siguientes notas son mis apuntes personales del proyecto, las cuales me ayudaron a entender, organizar, estructurar y documentar mis ideas durante el desarrollo de este.

**Nombre del proyecto:** _Clasificador de niveles de estrés a partir de señales fisiológicas (Stress Level Classification API)_

**Categoría:** Salud y bienestar.

**Motivación:** Caso realista, alineado con mis intereses en bienestar, cognición, bioseñales y multimodalidad.

**Objetivo:** Desarrollar un sistema que clasifique el nivel de estrés de una persona (por ejemplo: bajo, medio, alto) a partir de señales fisiológicas (como frecuencia cardíaca, GSR, temperatura) usando un modelo de ML, que luego se despliegue como una **API REST** lista para producción.

**Dataset:** WESAD dataset (Wearable Stress and Affect Detection)

- **Decripción:** 
    WESAD contiene datos de 15 sujetos obtenidos durante un estudio de laboratorio sobre estrés y estados afectivos, mientras llevaban puestos sensores fisiológicos y de movimiento.

    Las modalidades de sensores incluidas son las siguientes: 
    - Pulso de volumen sanguíneo
    - Electrocardiograma
    - Actividad electrodérmica
    - Electromiograma 
    - Respiración
    - Temperatura corporal 
    - Aceleración en tres ejes

    Además, el conjunto de datos salva la distancia entre estudios de laboratorio previos sobre el estrés y las emociones, al incluir tres estados afectivos diferentes (neutral, estrés y diversión).

    Asimismo, en el conjunto de datos se incluyen los autoinformes de los sujetos, obtenidos mediante varios cuestionarios establecidos. 

- **Clases:** neutral, estrés, diversión

- **URL:** https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

- **Cita completa:** Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection", ICMI 2018, Boulder, USA, 2018.

## Componentes del proyecto 

```mermaid
graph TD
  A[1 <br> Entendiendo el dataset] --> B[ 2 <br> Ingesta y Preprocesamiento de datos]
  B --> C[3 <br> Entrenamiento]
  C --> D[4 <br> Construcción de una API]
  D --> F[5 <br> Empaquetado y despliegue]
  F --> G[6 <br> Documentación + Monitoreo básico]
```  

## 1. Entendiendo el dataset 

WESAD (Wearable Stress and Affect Detection) es un dataset multimodal para detectar estrés y estados afectivos a partir de datos fisiológicos recolectados con dispositivos portables.

- 17 sujetos participaron, pero solo 15 tienen datos útiles (S1 y S12 se descartan).
- Cada sujeto tiene su propia carpeta

Dentro de cada carpeta se encuentra:
| Archivo           | Contenido                                                               |
| ----------------- | ----------------------------------------------------------------------- |
| `SX_readme.txt`   | Info específica del sujeto (calidad de datos, notas)                    |
| `SX_quest.csv`    | Horario del protocolo + respuestas de cuestionarios                     |
| `SX_respiban.txt` | Datos crudos del dispositivo **de pecho (RespiBAN)**                    |
| `SX_E4_Data.zip`  | Datos crudos del dispositivo **de muñeca (Empatica E4)**                |
| `SX.pkl`          | Diccionario con **datos ya sincronizados + etiquetas** |

### ¿Qué contiene SX.pkl?

Es un diccionario con tres claves principales:

```python
{
  'subject': 'S2',
  'signal': {
      'chest': {...},   # RespiBAN: ACC, ECG, EDA, EMG, RESP, TEMP
      'wrist': {...}    # E4: ACC, BVP, EDA, TEMP
  },
  'label': array([...])  # Array sincronizado con etiquetas por muestra
}
```
### Labels 

| Valor | Estado                    |
| ----- | ------------------------- |
| 0     | Transitorio / sin definir |
| 1     | **Baseline (neutral)**        |
| 2     | **Estrés**                    |
| 3     | **Diversión (amusement)**     |
| 4     | Meditación                |
| 5–7   | Ignorar                   |

### Dispositivos usados y señales obtenidas 

**RespiBAN (pecho):**

- Alta frecuencia (700 Hz)
- Modalidades:
    - ECG (Electrocardiogram)
    - EDA (Electrodermal Activity)
    - EMG (Electromyogram)
    - RESP (Respiration)
    - TEMP (Body Temperature)
    - ACC (Accelerometer (3-axis))

**Empatica E4 (muñeca):**

Muestras a frecuencias distintas:
- ACC (32 Hz)
- BVP (Blood Volume Pulse 64 Hz)
- EDA y TEMP (4 Hz)

El resampleado uniformiza la tasa de datos, haciendo que todas las señales tengan el mismo número de muestras por segundo, facilitando su procesamiento. Ya está todo resampleado en el `.pkl` en este caso. 

### Cuestionarios (SX_quest.csv)

- Autoevaluaciones PANAS, STAI, SAM, SSSQ
- Horarios de cada condición (líneas 2–4)

La sincronización uniformiza la línea de tiempo, garantizando que todos los datos estén perfectamente alineados en el tiempo. Por ahora podemos no usar estos archivos, ya que las etiquetas del protocolo ya están sincronizadas en el `.pkl`

## 2. Ingesta y preprocesamiento de datos 
### Exploración inicial  
Lo que se hizo fue:
- Cargar los datos del sujeto S2
  ```python 
  subject_id = "S2"
  subject_path = f"/kaggle/input/wesad-full-dataset/WESAD/{subject_id}/{subject_id}.pkl"
  with open(subject_path, 'rb') as file:
    data = pickle.load(file, encoding='latin1')
  ```
- Explorar las claves del diccionario, dimensiones y tipos de datos 
  ```
  Claves del diccionario: dict_keys(['signal', 'label', 'subject'])
  Modalidades disponibles: dict_keys(['chest', 'wrist'])
  Modalidades RespiBAN: dict_keys(['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'])
  Modalidades Empatica E4: dict_keys(['ACC', 'BVP', 'EDA', 'TEMP'])
  ```
  ```
  Tamaño de etiqueta: (4545100,)
  chest | ACC | shape: (4545100, 3)
  chest | ECG | shape: (4545100, 1)
  chest | EMG | shape: (4545100, 1)
  chest | EDA | shape: (4545100, 1)
  chest | Temp | shape: (4545100, 1)
  chest | Resp | shape: (4545100, 1)
  wrist | ACC | shape: (207776, 3)
  wrist | BVP | shape: (415552, 1)
  wrist | EDA | shape: (25972, 1)
  wrist | TEMP | shape: (25972, 1)
  ```

- Visualizar una señal fisiológica
  ![alt text](img/tempvsmuestras-S2.png)
- Ver la distribución de etiquetas
  ![alt text](img/etiquetas-S2.png)
### Preprocesamiento por ventanas y la extracción de características
Para esto vamos a trabajar solamente con las señales TEMP, EDA y BVP del dispositivo de muñeca y seguiremos unicamente con los datos asociados al sujeto 2. 
El flujo será más o menos el siguiente para cada señal:
```mermaid
  graph LR 
    A[Dividir la señal en ventanas temporales]-->B[Extraer estádisticas features por ventana]
    B-->C[Etiquetar cada ventana con su respectiva moda]
    C-->D[Eliminar las ventanas con etiquetas no relevantes]
```
> Esto se hace ya que no se entrenan modelos con una señal larga continua, sino con puntos de **datos representativos y compactos**. 
> Cada ventana es como una "instantánea" del estado fisiológico. Dentro de cada ventana, tomamos la etiqueta más frecuente (moda) y se la asignamos a esa ventana.

**Flujo completo en código**

1. Cargar archivo .pkl (S2)
2. Extraer señales y etiquetas
3. Definir ventanas
4. Extraer features TEMP, EDA, BVP
5. Convertir a arrays finales
6. Visualización exploratoria

al realizar el flujo anterior en código se obtiene las siguientes features y etiquetas asociadas al sujeto 2:

- **Features extraídos:** `(71, 18)`
- **Etiquetas válidas:** `(array([1, 2, 3]), array([38, 21, 12], dtype=int64))`

De la visualización exploratorio se obtiene:

**Distribución de clases en las ventanas extraídas**

- Clase 1 (baseline): Dominante, estado de reposo prolongado.
- Clase 2 (estrés) y Clase 3 (diversión): Menos frecuentes.

Nota: Existe un desvalance de clases. 

![alt text](img/distribucion-clases-s2.png)

**Pairplot de features por clase**
![alt text](img/pairplot-features-s2.png)

EDA: Parece separar bien estrés (2) de las otras clases.
El estrés genera mayor conductancia (más sudoración), por eso sube.

TEMP: Puede diferenciar baseline (1) vs diversión (3): la temperatura sube o baja levemente.

BVP: Hay variabilidad, pero no se ve tan separable visualmente. Aun así puede aportar info al combinarse con otros features.

Matriz de correlación
![alt text](img/matriz-correlaciones-s2.png)
`temp_mean` está muy correlacionado con otras estadísticas de temperatura

## 3. Entrenamiento 
Para esta sección se trabajo de acuerdo al siguiente flujo 
```mermaid 
  graph LR
  A[Dividir datos en entrenamiento y prueba]--> B[Entrenar modelo base]
  B-->C[Predicciones y evaluación]
  C-->D[Matriz de confusión]
```
Se entrenó un modelo base de clasificación usando Random Forest sobre las señales de muñeca (TEMP, EDA y BVP) del sujeto 2, obteniendo los siguientes resultados: 

![alt text](img/matriz-confusion-s2.png)

Es decir todas las clases fueron clasificadas correctamente, con un accuracy de 100%. Estos resultados deben interpretarse con cautela ya que las muestras son muy pocas (15 muestras). Además, existe la posibilidad de un sobre ajuste. Otro punto importante es que este modelo solo ha sido entrenado únicamente con datos del sujeto dos, por lo que aún no se ha generalizado y el  modelo podría estar reconociendo el estilo personal del sujeto y no la emoción en general. 