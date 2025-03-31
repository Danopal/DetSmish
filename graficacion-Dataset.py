import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Cargar el dataset TF-IDF
df_tfidf = pd.read_csv('tfidf_matrix_dataset.csv')

# Separar características y etiquetas
X = df_tfidf.iloc[:, :-1]  # Todas las columnas excepto la última
y = df_tfidf.iloc[:, -1]   # Última columna (etiquetas)

# Ver estructura del dataset
print(f"Dimensiones del dataset: {X.shape}")
print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
print("\nDistribución de etiquetas:")
print(y.value_counts())

# Gráfico de distribución de clases
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Etiquetas')
plt.xlabel('Etiqueta')
plt.ylabel('Cantidad de Muestras')
plt.xticks(rotation=0)
plt.show()
# Gráfico de distribución de clases
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Etiquetas')
plt.xlabel('Etiqueta')
plt.ylabel('Cantidad de Muestras')
plt.xticks(rotation=0)
plt.show()

# Obtener las 20 palabras con mayor peso promedio TF-IDF
mean_tfidf = X.mean(axis=0).sort_values(ascending=False)[:20]

plt.figure(figsize=(12, 6))
mean_tfidf.plot(kind='barh', color='teal')
plt.title('Top 20 Palabras con Mayor Peso TF-IDF Promedio')
plt.xlabel('Peso TF-IDF Promedio')
plt.ylabel('Palabra')
plt.gca().invert_yaxis()  # Invertir eje Y para mostrar la mayor en la parte superior
plt.show()

# Obtener las 10 palabras más importantes por categoría
unique_labels = y.unique()
top_n = 10

plt.figure(figsize=(15, 8))
for i, label in enumerate(unique_labels, 1):
    plt.subplot(1, len(unique_labels), i)
    label_indices = y[y == label].index
    mean_tfidf = X.iloc[label_indices].mean(axis=0).sort_values(ascending=False)[:top_n]
    mean_tfidf.plot(kind='barh', color='skyblue')
    plt.title(f'Top {top_n} - {label}')
    plt.xlabel('Peso TF-IDF')
    if i == 1:
        plt.ylabel('Palabra')
    else:
        plt.ylabel('')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Seleccionar las 20 características más importantes
top_features = X.mean(axis=0).sort_values(ascending=False).index[:20]
X_top = X[top_features]

# Calcular matriz de correlación
corr_matrix = X_top.corr()

# Graficar heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            xticklabels=top_features, yticklabels=top_features)
plt.title('Mapa de Calor de Correlación entre las 20 Características más Importantes')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Reducción a 3D con PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)

# Crear figura 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                    c=y.astype('category').cat.codes, cmap='viridis', alpha=0.6)

# Configurar leyenda
legend_labels = {code: label for code, label in enumerate(y.unique())}
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=plt.cm.viridis(code/len(legend_labels)), 
                      markersize=10) for code in legend_labels]
ax.legend(handles, legend_labels.values(), title='Etiquetas')

# Configurar ejes
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('Visualización 3D del Dataset TF-IDF')
plt.show()