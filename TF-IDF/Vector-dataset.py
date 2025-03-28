import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

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
        
        Parámetros:
        - remove_punct: eliminar puntuación
        - remove_numbers: eliminar números
        - remove_stopwords: eliminar stopwords
        - lemmatize: aplicar lematización
        - remove_emails: eliminar direcciones de email
        - remove_urls: eliminar URLs
        - remove_html: eliminar tags HTML
        - remove_special_chars: eliminar caracteres especiales
        - language: idioma para stopwords y lematización
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
        
        # Inicializar herramientas según configuración
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def clean_text(self, text):
        """Aplica todas las transformaciones de limpieza al texto"""
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Eliminar URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar HTML tags
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        # Eliminar caracteres especiales
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Eliminar puntuación
        if self.remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Eliminar números
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenización
        words = word_tokenize(text)
        
        # Eliminar stopwords
        if self.remove_stopwords and self.stop_words:
            words = [w for w in words if w not in self.stop_words]
        
        # Lematización
        if self.lemmatize and self.lemmatizer:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)

def load_data(filepath):
    """Carga el dataset desde un archivo CSV"""
    df = pd.read_csv(filepath)
    
    # Verificar columnas
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

def create_and_save_tfidf(df, save_path='tfidf_vectorizer.pkl', **tfidf_params):
    """
    Crea y guarda el vectorizador TF-IDF
    
    Parámetros:
    - df: DataFrame con los datos ya limpios
    - save_path: ruta para guardar el vectorizador
    - tfidf_params: parámetros para el TfidfVectorizer
    
    Retorna:
    - X_tfidf: matriz de características vectorizadas
    - vectorizer: el vectorizador entrenado
    """
    # Configuración por defecto (puede ser sobrescrita por tfidf_params)
    default_params = {
        'max_features': 10000,
        'min_df': 5,
        'max_df': 0.7,
        'ngram_range': (1, 2),
        'stop_words': None,  # Ya eliminamos stopwords en la limpieza
        'norm': 'l2',
        'analyzer': 'word'
    }
    
    # Combinar parámetros por defecto con los proporcionados
    params = {**default_params, **tfidf_params}
    
    # Crear y entrenar el vectorizador
    vectorizer = TfidfVectorizer(**params)
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    
    # Guardar el vectorizador
    with open(save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nVectorizador TF-IDF guardado en {save_path}")
    print(f"Vocabulario size: {len(vectorizer.get_feature_names_out())}")
    
    return X_tfidf, vectorizer

def save_clean_data(df, save_path='clean_data.csv'):
    """Guarda los datos limpios para uso futuro"""
    df.to_csv(save_path, index=False)
    print(f"\nDatos limpios guardados en {save_path}")

def main():
    # Configuración
    DATA_PATH = 'Datasets/train.csv'
    VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
    CLEAN_DATA_PATH = 'clean_data.csv'
    
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
        language='spanish'  # Asumiendo que el texto está en español
    )
    
    # 3. Limpiar texto
    print("Limpiando texto...")
    df['clean_text'] = df['Mensaje'].apply(cleaner.clean_text)
    
    # 4. Mostrar ejemplos y estadísticas
    show_examples(df)
    
    # 5. Crear y guardar vectorizador TF-IDF
    print("\nCreando vectorizador TF-IDF...")
    X_tfidf, vectorizer = create_and_save_tfidf(
        df,
        save_path=VECTORIZER_PATH,
        max_features=15000,  # Puedes ajustar este parámetro
        ngram_range=(1, 2)   # Unigramas y bigramas
    )
    
    # 6. Guardar datos limpios
    save_clean_data(df, CLEAN_DATA_PATH)
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()