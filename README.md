# Clasificador de estrés a partir de señales fisiológicas - Stress Classification

Proyecto de clasificación de estrés a partir de señales fisiológicas usando Machine Learning. El objetivo es detectar si una persona se encuentra en un estado de **estrés** a partir de datos recolectados con dispositivos portables. 

**Categoría:** Salud y bienestar.

**Motivación:** Caso realista, alineado con mis intereses en bienestar, cognición, bioseñales y multimodalidad.

---

## Objetivo

Desarrollar un sistema de machine learning capaz de detectar si una persona está en estado de estrés usando señales fisiológicas. En una primera versión, se pretende exponer el modelo a través de una API REST mínima que reciba *features* procesadas y devuelva la probabilidad de estrés junto con la clase predicha.



---

## Dataset: WESAD

- **Nombre completo:** *Wearable Stress and Affect Detection*.  
- **Sujetos:** 15 (S1 y S12 descartados por datos incompletos).  
- **Modalidades de sensores:**
  - Pulso de volumen sanguíneo (BVP)  
  - Electrocardiograma (ECG)  
  - Actividad electrodérmica (EDA)  
  - Electromiograma (EMG)  
  - Respiración (RESP)  
  - Temperatura corporal (TEMP)  
  - Aceleración en tres ejes (ACC)  

- **Estados medidos:**  
  - 1 = Baseline (neutral)  
  - 2 = Estrés  
  - 3 = Diversión  
  - (otros: transitorio/meditación, ignorados en este proyecto)  

- **Fuente:** [WESAD dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)  
- **Referencia:** Philip Schmidt et al. *Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection*, ICMI 2018.

---

## Etapas del proyecto

El proyecto se organiza en etapas incrementales:

1. **Etapa 1. Modelo base (1 sujeto)**  
   - Exploración inicial con S2.  
   - Preprocesamiento por ventanas (60s con solape de 30s).  
   - Features: estadísticas simples de TEMP, EDA y BVP.  
   - Entrenamiento con Random Forest. **Accuracy alto (100%) pero sobreajuste al sujeto.**

2. **Etapa 2. Generalización entre sujetos (LOSO)**  
   - Inclusión de múltiples sujetos.  
   - Normalización por sujeto (niveles basales).  
   - Validación **Leave-One-Subject-Out (LOSO)**.  
   - Random Forest con `class_weight=balanced`.  
   - Resultados:  
     - Accuracy global: **74%**  
     - Macro-F1: **0.64**  
     - Estrés se reconoce bien (F1 ≈ 0.73).  
     - Diversión es difícil de distinguir (F1 ≈ 0.36).  

3. **Etapa 3. Reencuadre binario (estrés vs no-estrés)**  
   - Próxima etapa: simplificar a binario para mejorar recall de estrés.  
   - Incorporar nuevas features (ej. HRV, dinámicas de EDA).  

4. **Etapa 4. API mínima**  
   - Endpoint `/predict`.  
   - Entrada: features procesadas.  
   - Salida: `prob_stress`, `label`.  

---

## Estructura del repositorio

```
.
├───code
    ├───etapa1
    |       01_Modelo_Base_Sujeto_Unico.ipynb
    │
    └───etapa2
            01_Carga_Exploracion_Multisujeto.ipynb
            02_Extraccion_Features_Multisujeto.ipynb
            03_Normalizacion_Evaluacion_LOSO.py
            features_raw.csv
            labels_por_sujeto_0a7.csv
            labels_por_sujeto_123.csv
            loso_report.txt
            loso_results.csv
└───docs
    └───notes.md 
    └───img
```

## Contacto 
Si te interesa el proyecto puedes escribirme al email: [nbaezhuber@gmail.com](mailto:nbaezhuber@gmail.com)