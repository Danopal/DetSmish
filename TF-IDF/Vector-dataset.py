import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize

# Descargar recursos de NLTK (solo primera vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, 
                 remove_punct=True, 
                 remove_numbers=True, 
                 remove_stopwords=True, 
                 stem_words=False, 
                 lemmatize_words=True,
                 language='english'):
        """
        Inicializa el preprocesador de texto con configuraciones personalizables
        
        Parámetros:
        - remove_punct: bool, eliminar puntuación
        - remove_numbers: bool, eliminar números
        - remove_stopwords: bool, eliminar stopwords
        - stem_words: bool, aplicar stemming
        - lemmatize_words: bool, aplicar lematización
        - language: str, idioma para stopwords y lematización
        """
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.lemmatize_words = lemmatize_words
        self.language = language
        
        # Inicializar herramientas según configuración
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else None
        self.stemmer = SnowballStemmer(language) if stem_words else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize_words else None
        
    def preprocess_text(self, text):
        """Aplica todas las transformaciones de limpieza al texto"""
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
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
        
        # Stemming
        if self.stem_words and self.stemmer:
            words = [self.stemmer.stem(w) for w in words]
        
        # Lematización
        if self.lemmatize_words and self.lemmatizer:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)

def load_and_preprocess_data(filepath, text_column, label_column, preprocessor):
    """
    Carga y preprocesa los datos del CSV
    
    Parámetros:
    - filepath: ruta al archivo CSV
    - text_column: nombre de la columna con el texto
    - label_column: nombre de la columna con las etiquetas
    - preprocessor: instancia de TextPreprocessor
    
    Retorna:
    - textos preprocesados y etiquetas
    """
    df = pd.read_csv(filepath)
    
    # Verificar columnas
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columnas '{text_column}' o '{label_column}' no encontradas en el CSV")
    
    # Preprocesar texto
    df['processed_text'] = df[text_column].apply(preprocessor.preprocess_text)
    
    return df['processed_text'].values, df[label_column].values

def create_tfidf_vectorizer(max_features=None, 
                          min_df=1, 
                          max_df=1.0, 
                          ngram_range=(1,1), 
                          use_idf=True, 
                          norm='l2',
                          stop_words=None,
                          analyzer='word'):
    """
    Crea y configura un vectorizador TF-IDF
    
    Parámetros ajustables:
    - max_features: número máximo de features (vocabulario)
    - min_df: ignorar términos con frecuencia menor a este valor (absoluto) o proporción (si <1)
    - max_df: ignorar términos con frecuencia mayor a este valor (proporción)
    - ngram_range: rango de n-grams a considerar
    - use_idf: habilitar/deshabilitar IDF
    - norm: norma para normalización ('l1', 'l2' o None)
    - stop_words: lista de stopwords o 'english' para usar las incorporadas
    - analyzer: 'word', 'char' o 'char_wb'
    """
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        use_idf=use_idf,
        norm=norm,
        stop_words=stop_words,
        analyzer=analyzer
    )

def train_svm(X_train, y_train, 
             C=1.0, 
             kernel='rbf', 
             gamma='scale', 
             class_weight=None, 
             random_state=None):
    """
    Entrena un clasificador SVM con parámetros configurables
    
    Parámetros:
    - C: parámetro de regularización
    - kernel: 'linear', 'poly', 'rbf', 'sigmoid'
    - gamma: coeficiente para kernels no lineales
    - class_weight: balanceo de clases (None, 'balanced' o dict)
    - random_state: semilla para reproducibilidad
    """
    clf = svm.SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
        probability=True
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y muestra métricas"""
    y_pred = model.predict(X_test)
    print("Exactitud:", accuracy_score(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

def optimize_parameters(X, y):
    """Optimización de hiperparámetros con GridSearchCV"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', svm.SVC())
    ])
    
    parameters = {
        'tfidf__max_features': [5000, 10000, None],
        'tfidf__ngram_range': [(1,1), (1,2)],
        'tfidf__norm': ['l1', 'l2'],
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    print("\nMejores parámetros encontrados:")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_

def main():
    # Configuración personalizable
    CSV_PATH = 'tu_dataset.csv'  # Cambiar por tu ruta
    TEXT_COLUMN = 'text'         # Nombre columna con texto
    LABEL_COLUMN = 'label'       # Nombre columna con etiquetas
    
    # 1. Configurar preprocesamiento
    preprocessor = TextPreprocessor(
        remove_punct=True,
        remove_numbers=True,
        remove_stopwords=True,
        stem_words=False,        # Generalmente lematización es mejor que stemming
        lemmatize_words=True,
        language='english'       # Cambiar según idioma del texto
    )
    
    # 2. Cargar y preprocesar datos
    X, y = load_and_preprocess_data(CSV_PATH, TEXT_COLUMN, LABEL_COLUMN, preprocessor)
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Configurar TF-IDF (parámetros ajustables)
    tfidf_vectorizer = create_tfidf_vectorizer(
        max_features=10000,      # Limitar vocabulario para eficiencia
        min_df=5,                # Ignorar términos muy raros
        max_df=0.7,              # Ignorar términos muy comunes
        ngram_range=(1,2),      # Unigramas y bigramas
        use_idf=True,
        norm='l2',
        stop_words='english',    # None si ya se eliminaron stopwords en preprocesamiento
        analyzer='word'
    )
    
    # 5. Vectorizar texto
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # 6. Entrenar SVM (parámetros ajustables)
    svm_classifier = train_svm(
        X_train_tfidf, y_train,
        C=1.0,
        kernel='linear',         # Kernel lineal suele funcionar bien con texto
        gamma='scale',
        class_weight='balanced'  # Útil si las clases están desbalanceadas
    )
    
    # 7. Evaluar modelo
    print("Evaluación con parámetros iniciales:")
    evaluate_model(svm_classifier, X_test_tfidf, y_test)
    
    # 8. Opcional: Optimización de parámetros (lento pero puede mejorar resultados)
    print("\nOptimizando parámetros...")
    best_model = optimize_parameters(X_train, y_train)
    
    # Evaluar modelo optimizado
    print("\nEvaluación con parámetros optimizados:")
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()