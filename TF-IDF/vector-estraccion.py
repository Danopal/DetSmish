import re
from joblib import load

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
    print(processed_text)
    return text_vector

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo .joblib generado por tu código
    tfidf_joblib_file = "vectorizador_tfidf.pkl"
    # Texto a vectorizar
    new_text = "'8:06 u 4660  +52 55 9711 7465  text mossage + sms today 8:08 0.5  netflix : problema de pago en su perfil!  verifica y actualiza tus datos antes del 27/03 aqui:  no y si qwertyouio sp asdfeghjklñ  zxcvbnm  123 (o) espacio intro  $ q"
    
    # Vectorizar el texto
    vectorized_result = vectorize_text(new_text, tfidf_joblib_file)
    print("Vector TF-IDF:", vectorized_result)