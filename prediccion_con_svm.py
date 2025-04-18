import joblib
import numpy as np
import pandas as pd
from ocr import post_proc  # Asegúrate de tener esta función que procesará la imagen

# Función para cargar el modelo SVM desde el archivo .pkl
def cargar_modelo(ruta_modelo):
    modelo = joblib.load(ruta_modelo)
    return modelo

# Función para hacer predicciones con el modelo cargado
def hacer_predicciones(modelo, datos_nuevos_df):
    predicciones = modelo.predict(datos_nuevos_df)
    
    predicciones_transformadas = ['no smishing' if pred == 'ham' else 'smishing' for pred in predicciones]
    
    return predicciones_transformadas

# Función para cargar el vectorizador TF-IDF desde el archivo .joblib
def load_tfidf_vectorizer(joblib_file):
    vectorizer = joblib.load(joblib_file)
    return vectorizer

# Función para preprocesar el texto
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar signos de puntuación
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios adicionales
    return text

# Función para vectorizar el texto y convertirlo a DataFrame con nombres de características
def vectorize_text(new_text, joblib_file):
    processed_text = preprocess_text(new_text)
    tfidf_vectorizer = load_tfidf_vectorizer(joblib_file)
    
    # Obtener la representación TF-IDF del texto
    text_vector = tfidf_vectorizer.transform([processed_text])
    
    # Convertir el resultado en un DataFrame con los nombres de características
    feature_names = tfidf_vectorizer.get_feature_names_out()
    text_vector_df = pd.DataFrame(text_vector.toarray(), columns=feature_names)
    
    return text_vector_df

# Ejemplo de uso
if __name__ == "__main__":
    # Rutas de los archivos
    tfidf_joblib_file = "vect_tfidf.pkl"  # Ruta del vectorizador TF-IDF
    svm_model_file = "svm_entrenado.pkl"  # Ruta del modelo SVM entrenado
    
    # Texto a vectorizar desde el OCR
    new_text = post_proc("imagenes-ocr/cap1.jpg")  
    
    # Vectorizar el texto
    vectorized_result_df = vectorize_text(new_text, tfidf_joblib_file)

    # Cargar el modelo SVM
    modelo_svm = cargar_modelo(svm_model_file)
    
    # Hacer predicciones
    predicciones = hacer_predicciones(modelo_svm, vectorized_result_df)
    
    # Mostrar la predicción
    print(f"Predicción del mensaje '{new_text}' es: {predicciones[0]}")