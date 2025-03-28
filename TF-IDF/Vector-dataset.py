import pandas as pd

# Cargar el dataset TF-IDF
df_tfidf = pd.read_csv('tfidf_matrix_dataset.csv')

# Las columnas son los términos y la última columna es la etiqueta
features = df_tfidf.iloc[:, :-1]
labels = df_tfidf['Etiqueta']