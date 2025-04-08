import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Cargar el dataset tfidf.csv
ruta_tfidf = 'tfidf.csv'
df_tfidf = pd.read_csv(ruta_tfidf)

# Eliminar filas donde la etiqueta es 'Etiqueta' (valor no válido)
df_tfidf = df_tfidf[df_tfidf['Etiqueta'] != 'Etiqueta']

# Separar las características (X) y la variable objetivo (y)
X = df_tfidf.drop(columns=['Etiqueta'])
y = df_tfidf['Etiqueta']

# Dividir en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir la grilla de parámetros para GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],  # Probar tanto con kernel lineal como RBF
    'gamma': ['scale', 'auto']    # Probar con dos opciones de gamma
}

# Crear el GridSearchCV para buscar los mejores parámetros
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)

# Entrenar el modelo usando GridSearchCV
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_svm = grid_search.best_estimator_

# Realizar predicciones con el mejor modelo
predicciones = best_svm.predict(X_test)

# Evaluar el rendimiento del modelo
informe = classification_report(y_test, predicciones, target_names=y.unique())
print(informe)

# Guardar el mejor modelo entrenado
joblib.dump(best_svm, 'svm_model_best.pkl')
# Imprimir los mejores parámetros encontrados por GridSearchCV
print("Mejores parámetros:", grid_search.best_params_)

# Imprimir el mejor rendimiento (exactitud) obtenido con los mejores parámetros
print("Mejor rendimiento obtenido:", grid_search.best_score_)

