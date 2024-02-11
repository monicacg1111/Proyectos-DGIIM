# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2023
Contenido:
    Ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.impute import KNNImputer
from math import floor
import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

datos = pd.read_csv('03_Datos_noviembre_2023_num.csv')
peso = 'ponde'

# Se pueden reemplazar los valores desconocidos por un número
# datos = datos.replace(np.NaN,0)

# O imputar, por ejemplo con la media (el impacto en clustering es distinto a clasificación)
'''
for col in datos:
    datos[col].fillna(datos[col].mean(), inplace=True)
'''

# Pero mejor si se imputa por vecinos más cercanos
#'''
imputer = KNNImputer(n_neighbors=3)
datos_norm = datos.apply(norm_to_zero_one)
datos_imputados_array = imputer.fit_transform(datos_norm)
datos_norm2 = pd.DataFrame(datos_imputados_array, columns = datos.columns)
datos = datos_norm2
#'''
     
'''
for col in datos:
   missing_count = sum(pd.isnull(datos[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

# Seleccionar casos, filtrando por sexo
subset = datos.loc[(datos['sexo']==1)] #solo mujeres (1==2 después de normalizar)

# Seleccionar variables de interés para clustering
# renombramos las variables por comodidad
subset=subset.rename(columns={"p7": "ideología", "edu": "educación", "cs": "clase social", "edad": "edad", "p1": "ahorro"})
usadas = ['ideología','educación','clase social','edad', 'ahorro']

n_var = len(usadas)
X = subset[usadas]

# eliminar outliers como aquellos casos fuera de 1.5 veces el rango intercuartil
'''
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X = X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)]
'''

# normalizamos
X_normal = X.apply(norm_to_zero_one)

print('----- Ejecutando k-Means',end='')
k_means = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=123456)
t = time.time()
cluster_predict = k_means.fit_predict(X_normal,subset[peso]) #se usa peso para cada objeto (factor de elevación)
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')

# Esto es opcional, el cálculo de Silhouette puede consumir mucha RAM.
# Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
muestra_silhouette = 0.2 if (len(X) > 10000) else 1.0
   
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhouette*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

# se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
size = size.sort_index()

for i,c in enumerate(size):
   print('%s: %5d (%5.2f%%)' % (i,c,100*c/len(clusters)))

k = len(size)
colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)

#'''
print("---------- Heatmap de centroides...")
centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

# se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

plt.figure()
centers.index += 1
plt.figure()
hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, annot_kws={"fontsize":18}, fmt='.3f')
hm.set_ylim(len(centers),0)
hm.figure.set_size_inches(15,15)
hm.figure.savefig("centroides.pdf")
centers.index -= 1
#'''

# se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)

#'''
print("---------- Scatter matrix...")
plt.figure()
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
X_kmeans['cluster'] += 1
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette=colors, plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
X_kmeans['cluster'] -= 1
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
sns_plot.fig.set_size_inches(15,15)
sns_plot.savefig("scatter.pdf")
plt.show()
#'''