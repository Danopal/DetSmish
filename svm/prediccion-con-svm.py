#programa para la predicción de un mensaje de smishing mediante un algortimo de clasificación 
# entrenado en  entrenamiento-svm.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Cargar datos
data = pd.read_csv("Datasets/train.csv")
X = data["Mensaje"]
y = data["Etiqueta"]

# Vectorización TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Dividir en train y test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)
from sklearn.svm import SVC

# Crear y entrenar el modelo SVM
model = SVC(kernel="linear", class_weight="balanced")  # Kernel lineal funciona bien para texto
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Mostrar reporte de clasificación
print(classification_report(y_test, y_pred))

# Matriz de confusión (opcional)
print(confusion_matrix(y_test, y_pred))

