# Proyectos-DGIIM
Repositorio donde iré subiendo algunos proyectos que he realizado durante la carrera. Se trata de varios proyectos sobretodo relacionados con el Análisis de Datos.

## Clustering CIS

Proyecto realizado en la asignatura *Inteligencia de Negocio* para extraer conclusiones sobre la opinión pública respecto al conflicto bélico entre Israel y Palestina. Se trata de una encuesta realizada por el CIS en noviembre de 2023, donde recoge la intención de voto de la
población española y su opinión respecto a varios asuntos de actualidad (conflicto de Israel y Palestina, crisis de inflación, cambio climático, migración...). Se puede consultar [aquí](https://elpais.com/espana/2023-11-06/consulte-todos-los-datos-internos-de-la-encuesta-de-el-pais-cuestionarios-cruces-y-respuestas-individuales.html).

Aparte de aplicar varios **algoritmos de Clustering** (tales como K-means, BIRCH, Mean Shift o DBSCAN), he llevado a cabo un preprocesamiento mediante técnicas de normalización e imputación.

El proyecto ha consistido en establecer varios casos de estudios donde he tomado en cada uno varias características (respuestas a la escuesta) y he estudiado la relación entre ellas. Entre estos casos de estudio, he estudiado la relación entre la **ideología** del encuestado, su **intención de voto** y su **opinión** sobre distintas acciones cometidas tanto por Israel como por Hamás. Para interpretar los resultados de los algoritmos me he servido de varias herramientas de **visualización**, como Heatmaps (mapas de calor), Scatter Matrix, MDS, etc.



## Clasificación Kaggle

Proyecto realizado en la asignatura *Inteligencia de Negocio* que ha consistido en participar en una [Competición de Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/). El objetivo de la competición fue el de **predecir el precio de una casa** dada una serie de características, tales como el tamaño, el tipo de calle, el número de habitaciones, etc. Dado un dataset de entrenamiento, había que entrenar un **clasificador** que fuera capaz de predecir con una precisión alta el precio de las casas dadas en un nuevo dataset de prueba.

Para resolver el problema, antes de nada apliqué de forma exhaustiva varias técnicas de **preprocesamiento**, como eliminación de outliers, selección de las características más relevantes, imputación de valores perdidos y etiquetado. Posteriormente apliqué varios **algoritmos de aprendizaje supervisado**, entre ellos, métodos basados en Árboles de decisión, Boosting y Ensemble. Además, realicé un **tuning automático** para ajustar los parámetros del método elegido y así conseguir la mejor eficacia posible. 

En cuanto a los resultados finales, el algoritmo que mejor resultado dio fue Catboost, dejándome en una posición bastante buena en el Ranking para ser mi primera competición. Al final resultó ser una experiencia muy enriquecedora e innovadora. 

![leaderboard2](C:\Users\monic\Documents\Proyectos-DGIIM\Clasificación Kaggle\img\leaderboard2.png)