# Análisis Comparativo de Técnicas de Vectorización de Texto en PLN

Este repositorio presenta un estudio empírico exhaustivo sobre estrategias de representación vectorial de texto para modelos de Procesamiento de Lenguaje Natural (PLN). El proyecto evalúa el impacto de la sintonización de hiperparámetros, el costo computacional del preprocesamiento, la escalabilidad en memoria y el rendimiento de métodos estadísticos frente a word embeddings densos.

## Arquitectura de los Experimentos

El código fuente estructura el análisis en cinco fases metodológicas:

1. **Optimización de Espacio Vectorial (Bag of Words):** Evaluación empírica del control de ruido y dimensionalidad mediante parámetros de frecuencia de documentos (`min_df`, `max_df`) y truncamiento de vocabulario (`max_features`).
2. **Análisis de Contexto Local (N-gramas):** Medición de la relación costo-beneficio (expansión de dimensionalidad vs. ganancia predictiva) al incorporar bigramas y trigramas.
3. **Carga Computacional del Preprocesamiento:** Benchmark de tiempo de ejecución y precisión predictiva comparando normalización estandarizada frente a lematización morfosintáctica utilizando arquitecturas de spaCy.
4. **Análisis de Dispersión y Complejidad Espacial:** Verificación del comportamiento de crecimiento del vocabulario (Ley de Heaps) y demostración de eficiencia asintótica en el uso de memoria RAM mediante matrices dispersas (formato CSR).
5. **Enfoques Estadísticos vs. Representaciones Densas:** Comparativa de rendimiento entre arquitecturas TF-IDF clásicas y embeddings semánticos preentrenados (FastText). La evaluación se segmenta según la separación de las clases (tópicos ortogonales vs. dominios semánticamente adyacentes).

## Hallazgos Principales

<img width="1589" height="593" alt="image" src="https://github.com/user-attachments/assets/0ce397f9-e437-479e-93b1-25d68b372f91" />


* **Sintonización de Frecuencias:** La aplicación rigurosa de umbrales inferiores y superiores (`min_df >= 2`, `max_df = 0.95`) permite una drástica reducción del tamaño del vocabulario mitigando la maldición de la dimensionalidad sin degradación del *Accuracy*.
* **Saturación por N-gramas:** La adición de trigramas genera una explosión combinatoria en el espacio de características que, en modelos lineales, tiende a sobreajustar o degradar la métrica objetivo. El rango `(1, 2)` demuestra ser empíricamente superior.
* **Eficiencia de Estructuras Dispersas:** En corpus de tamaño moderado a grande, la densidad de la matriz de características es inferior al 1%. Las estructuras Compressed Sparse Row (CSR) logran factores de compresión de memoria superiores a 100x frente a tensores densos estándar.
* **Paradigma de Modelado Final:** Para la clasificación de textos largos, las representaciones basadas en TF-IDF mantienen una competitividad robusta y sirven como baseline primario. Los embeddings densos (FastText promediado) resultan preferibles únicamente ante escenarios de textos cortos o alta sinonimia, pero a un mayor costo de inferencia.

## Stack Tecnológico

El entorno de experimentación fue desarrollado en Python utilizando el siguiente stack:

* **scikit-learn:** Extracción de características (CountVectorizer, TfidfVectorizer), modelado predictivo (Regresión Logística) y métricas de evaluación.
* **spaCy:** Procesamiento lingüístico avanzado y lematización.
* **Gensim:** Carga y manipulación de Word Embeddings (FastText subwords).
* **NumPy / SciPy:** Operaciones algebraicas subyacentes y gestión de matrices dispersas.
* **Pandas & Matplotlib:** Estructuración de resultados y generación de gráficas analíticas.
