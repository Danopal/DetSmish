import pandas as pd
import re
import string
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from scipy.sparse import csr_matrix

# Descargar recursos de NLTK (solo primera vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextCleaner:
    def __init__(self, 
                 remove_punct=True,
                 remove_numbers=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 remove_emails=True,
                 remove_urls=True,
                 remove_html=True,
                 remove_special_chars=True,
                 language='spanish'):
        """
        Inicializa el limpiador de texto con configuraciones personalizables
        """
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_emails = remove_emails
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.language = language
        
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def clean_text(self, text):
        """Aplica todas las transformaciones de limpieza al texto"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        
        if self.remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        words = word_tokenize(text)
        
        if self.remove_stopwords and self.stop_words:
            words = [w for w in words if w not in self.stop_words]
        
        if self.lemmatize and self.lemmatizer:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)

def load_data(filepath):
    """Carga el dataset desde un archivo CSV"""
    df = pd.read_csv(filepath)
    
    if 'Mensaje' not in df.columns or 'Etiqueta' not in df.columns:
        raise ValueError("El dataset debe contener columnas 'Mensaje' y 'Etiqueta'")
    
    return df

def show_examples(df, n=3):
    """Muestra ejemplos del dataset antes y después de la limpieza"""
    print("\n=== Ejemplos del dataset original ===")
    print(df[['Mensaje', 'Etiqueta']].head(n))
    
    print("\n=== Ejemplos del dataset limpio ===")
    print(df[['clean_text', 'Etiqueta']].head(n))
    
    print("\n=== Estadísticas descriptivas ===")
    print(f"Número total de mensajes: {len(df)}")
    print(f"Distribución de etiquetas:\n{df['Etiqueta'].value_counts()}")
    print(f"Longitud promedio del texto: {df['clean_text'].apply(len).mean():.1f} caracteres")

def create_tfidf_matrix(df, **tfidf_params):
    """
    Crea la matriz TF-IDF y devuelve el vectorizador y la matriz
    
    Parámetros:
    - df: DataFrame con los datos ya limpios
    - tfidf_params: parámetros para el TfidfVectorizer
    
    Retorna:
    - X_tfidf: matriz de características vectorizadas (sparse)
    - vectorizer: el vectorizador entrenado
    - feature_names: nombres de las características (palabras)
    """
    default_params = {
        'max_features': 10000,
        'min_df': 5,
        'max_df': 0.7,
        'ngram_range': (1, 2),
        'stop_words': None,
        'norm': 'l2',
        'analyzer': 'word'
    }
    
    params = {**default_params, **tfidf_params}
    vectorizer = TfidfVectorizer(**params)
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    
    return X_tfidf, vectorizer, vectorizer.get_feature_names_out()

def save_tfidf_dataset(X_tfidf, feature_names, labels, save_path='tfidf_dataset.csv'):
    """
    Guarda la matriz TF-IDF como un nuevo dataset CSV
    
    Parámetros:
    - X_tfidf: matriz sparse TF-IDF
    - feature_names: nombres de las características
    - labels: etiquetas originales
    - save_path: ruta para guardar el archivo
    """
    # Convertir la matriz sparse a DataFrame
    df_tfidf = pd.DataFrame.sparse.from_spmatrix(X_tfidf, columns=feature_names)
    
    # Añadir las etiquetas originales
    df_tfidf['Etiqueta'] = labels.values
    
    # Guardar el dataset
    df_tfidf.to_csv(save_path, index=False)
    print(f"\nDataset TF-IDF guardado en {save_path}")
    print(f"Dimensiones: {df_tfidf.shape[0]} muestras x {df_tfidf.shape[1]} características")

def save_artifacts(vectorizer, df_clean, vectorizer_path='tfidf_vectorizer.pkl', clean_data_path='clean_data.csv'):
    """Guarda el vectorizador y los datos limpios"""
    # Guardar vectorizador
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"\nVectorizador TF-IDF guardado en {vectorizer_path}")
    
    # Guardar datos limpios
    df_clean.to_csv(clean_data_path, index=False)
    print(f"Datos limpios guardados en {clean_data_path}")

def main():
    # Configuración
    DATA_PATH = 'Datasets/train.csv'
    VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
    CLEAN_DATA_PATH = 'clean_data.csv'
    TFIDF_DATASET_PATH = 'tfidf_matrix_dataset.csv'
    
    # 1. Cargar datos
    print("Cargando datos...")
    df = load_data(DATA_PATH)
    
    # 2. Configurar limpiador de texto
    cleaner = TextCleaner(
        remove_punct=True,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True,
        remove_emails=True,
        remove_urls=True,
        remove_html=True,
        remove_special_chars=True,
        language='spanish'
    )
    
    # 3. Limpiar texto
    print("Limpiando texto...")
    df['clean_text'] = df['Mensaje'].apply(cleaner.clean_text)
    
    # 4. Mostrar ejemplos y estadísticas
    show_examples(df)
    
    # 5. Crear matriz TF-IDF
    print("\nCreando matriz TF-IDF...")
    X_tfidf, vectorizer, feature_names = create_tfidf_matrix(
        df,
        max_features=15000,
        ngram_range=(1, 2)
    )
    
    # 6. Guardar dataset TF-IDF
    save_tfidf_dataset(X_tfidf, feature_names, df['Etiqueta'], TFIDF_DATASET_PATH)
    
    # 7. Guardar artefactos (vectorizador y datos limpios)
    save_artifacts(vectorizer, df, VECTORIZER_PATH, CLEAN_DATA_PATH)
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()