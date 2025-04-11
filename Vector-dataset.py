import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Descargar recursos de NLTK (solo primera vez)
nltk.download('punkt')
nltk.download('stopwords')

# Función para limpiar el texto
def clean_text(text, remove_punct=True, remove_numbers=True, remove_stopwords=True, lemmatize=True, language='spanish'):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Tokenizar
    words = word_tokenize(text)
    
    if remove_stopwords:
        stop_words = set(stopwords.words(language))
        words = [w for w in words if w not in stop_words]
    
    # Lematización básica (puedes sustituir por un lematizador si lo deseas)
    if lemmatize:
        words = [w if w not in string.punctuation else "" for w in words]  # Esto es solo un ejemplo simple de lematización

    return ' '.join(words)

# Función para cargar datos y validación de columnas
def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'Mensaje' not in df.columns or 'Etiqueta' not in df.columns:
        raise ValueError("El dataset debe contener las columnas 'Mensaje' y 'Etiqueta'")
    return df

# Función para mostrar ejemplos de los datos
def show_examples(df, n=3):
    print("\n=== Ejemplos del dataset original ===")
    print(df[['Mensaje', 'Etiqueta']].head(n))
    
    print("\n=== Ejemplos del dataset limpio ===")
    print(df[['clean_text', 'Etiqueta']].head(n))

# Función para crear la matriz TF-IDF
def create_tfidf_matrix(df, **tfidf_params):
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    return X_tfidf, vectorizer, vectorizer.get_feature_names_out()

# Función para guardar la matriz TF-IDF
def save_tfidf_dataset(X_tfidf, feature_names, labels, save_path='train_tfidf.csv'):
    df_tfidf = pd.DataFrame.sparse.from_spmatrix(X_tfidf, columns=feature_names)
    df_tfidf['Etiqueta'] = labels.values
    df_tfidf.to_csv(save_path, index=False, compression='gzip')
    print(f"\nDataset TF-IDF guardado en {save_path}")
    print(f"Dimensiones: {df_tfidf.shape[0]} muestras x {df_tfidf.shape[1]} características")

# Función para guardar el vectorizador
def save_artifacts(vectorizer, df_clean, vectorizer_path='tfidf_vectorizer.pkl', clean_data_path='clean_data.csv'):
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nVectorizador TF-IDF guardado en {vectorizer_path}")
    df_clean.to_csv(clean_data_path, index=False)
    print(f"Datos limpios guardados en {clean_data_path}")

# Función principal
def main():
    DATA_PATH = 'Datasets/train.csv'
    VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
    CLEAN_DATA_PATH = 'clean_data.csv'
    TFIDF_DATASET_PATH = 'train_tfidf.csv'
    
    # 1. Cargar datos
    print("Cargando datos...")
    df = load_data(DATA_PATH)
    
    # 2. Limpiar texto
    print("Limpiando texto...")
    df['clean_text'] = df['Mensaje'].apply(lambda x: clean_text(x, remove_punct=True, remove_numbers=True, remove_stopwords=True, lemmatize=True))
    
    # 3. Mostrar ejemplos y estadísticas
    show_examples(df)
    
    # 4. Crear matriz TF-IDF
    print("\nCreando matriz TF-IDF...")
    X_tfidf, vectorizer, feature_names = create_tfidf_matrix(df, max_features=15000, ngram_range=(1, 2))
    
    # 5. Guardar dataset TF-IDF
    save_tfidf_dataset(X_tfidf, feature_names, df['Etiqueta'], TFIDF_DATASET_PATH)
    
    # 6. Guardar artefactos (vectorizador y datos limpios)
    save_artifacts(vectorizer, df, VECTORIZER_PATH, CLEAN_DATA_PATH)
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()
