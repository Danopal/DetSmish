import re
from joblib import load
from ocr import post_proc  # Importamos la función port_proc de ocr.py


# Función para preprocesar el texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    # Eliminar signos de puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Cargar el vectorizador TF-IDF desde el archivo guardado con joblib
def load_tfidf_vectorizer(joblib_file):
    vectorizer = load(joblib_file)
    return vectorizer

# Procesar y vectorizar el texto
def vectorize_text(new_text, joblib_file):
    # Preprocesar el texto
    processed_text = preprocess_text(new_text)
    
    # Cargar el vectorizador TF-IDF
    tfidf_vectorizer = load_tfidf_vectorizer(joblib_file)
    
    # Vectorizar el texto preprocesado
    text_vector = tfidf_vectorizer.transform([processed_text])
    
    # Convertir la matriz dispersa en una matriz densa
    text_vector_dense = text_vector.toarray()  # Esto convierte la matriz dispersa en una matriz densa
    
    return text_vector_dense


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo .joblib generado por tu código
    tfidf_joblib_file = "vect_tfidf.pkl"
    # Texto a vectorizar
    new_text = post_proc("imagenes-ocr/cap3.jpg")
    
    # Vectorizar el texto
    vectorized_result = vectorize_text(new_text, tfidf_joblib_file)
    print("Vector TF-IDF:", vectorized_result)