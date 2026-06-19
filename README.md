# tfm-codigos

# Evaluación Robusta de Modelos SVM Multiclase
Este repositorio contiene el código fuente y los conjuntos de datos utilizados para el Trabajo de Fin de Máster (TFM) titulado **"Análisis de modelos de machine learning para problemas de clasificación multiclase"**, desarrollado por **Ana Marta Oliveira dos Santos** (Junio 2026).

## Descripción del Proyecto
El objetivo de este trabajo es analizar y comparar empíricamente las diferentes extensiones multiclase de las Máquinas de Vectores de Soporte (SVM), contrastando los enfoques basados en descomposición binaria frente a las formulaciones de optimización global simultánea. 

El repositorio se divide en dos bloques prácticos correspondientes a los capítulos de la memoria:

1. **Capítulo 2 - Ilustraciones:** Implementaciones propias desde cero en el espacio bidimensional para ilustrar la teoría de las SVM, abarcando el margen duro, margen suave, el truco del kernel (RBF) y las estrategias multiclase (*One-versus-One*, *One-versus-Rest* y *All-together*).
2. **Capítulo 3 - Experimentación Multiescenario:** Un entorno de evaluación paralelizado (`parfor`) que somete a los diferentes modelos (OvA, OvO, Weston-Watkins, Crammer-Singer) a escenarios base y a escenarios con inyección de ruido estocástico (en etiquetas y características) sobre cuatro conjuntos de datos reales. Incluye validación cruzada y evaluación estadística mediante el test de Wilcoxon.

---

## Estructura del Repositorio
La arquitectura del proyecto está organizada de la siguiente manera:
* **`Capitulo2/`**: Scripts creados para explicar la formulación matemática y generar las fronteras de decisión teóricas.
* **`Capitulo3/`**: Núcleo experimental del TFM.
  * `Script_Principal.m`: Script maestro que orquesta la validación cruzada, el control de *data leakage*, la inyección de ruido, la paralelización y la exportación de resultados.
  * `medidas_completas.m`: Función propia para el cálculo de métricas macro-promediadas (Accuracy, Balanced Accuracy, Precision, Recall, F1-Score).
  * **`Dataset_Multiclase/`**: Conjuntos de datos preprocesados y normalizados (*Iris, Hayes-Roth, Wine, Glass*) obtenidos del repositorio UCI Machine Learning.
  * **`Multi_SVM/`**: Funciones base de predicción multiclase adaptadas para el proyecto.
  * **`SVM/`**: Herramientas matemáticas puras (optimizadores cuadráticos y funciones de Kernel).

---

## Requisitos y Ejecución
* **Software:** MATLAB con el *Parallel Computing Toolbox* instalado.
* **Dependencia externa:** Es estrictamente necesario tener instalado y configurado en el *Path* el paquete de optimización convexa **CVX** (http://cvxr.com/cvx/) utilizando el *solver* SeDuMi.
* **Instrucciones de Ejecución:**
  1. Clonar este repositorio y añadir la carpeta raíz y todas sus subcarpetas al *Path* de MATLAB.
  2. Para reproducir las figuras teóricas, ejecutar cualquier script dentro de la carpeta `Capitulo2/`.
  3. Para reproducir la experimentación principal, ejecutar el archivo `Capitulo3/Script_Principal.m`. Al finalizar, se generará automáticamente un archivo Excel (`Resultados_TFM_Completo.xlsx`) con las métricas y el análisis estadístico.

---

## Autoría y Agradecimientos

**Trabajo de la autora:**
La arquitectura del flujo experimental, el sistema de validación cruzada paralelizado, el mecanismo de inyección de ruido estocástico, la función de métricas (`medidas_completas.m`) y todos los scripts ilustrativos del `Capitulo2/` han sido desarrollados íntegramente por la autora del TFM.

**Código proporcionado:**
Se agradece a los directores del proyecto por proporcionar los conjuntos de datos preprocesados y el código base de las herramientas matemáticas subyacentes ubicadas en las carpetas `Capitulo3/Multi_SVM/` y `Capitulo3/SVM/`. Estos *scripts* son implementaciones matemáticas base proporcionadas para la resolución cuadrática de los modelos y se incluyen en este repositorio para garantizar la transparencia y la total reproducibilidad de los experimentos documentados en la memoria.
