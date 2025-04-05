import pandas as pd
import re 
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



# Limpieza del texto
def preprocesamiento(texto, remove_punct=True, remove_numbers=True, remove_stopwords=True, lemmatize=True, language='spanish'):
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower()

    if remove_punct:
        texto = texto.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        texto = re.sub(r'\d+', '', texto)

    # Tokenizar
    palabras = word_tokenize(texto)
    
    if remove_stopwords:
        stop_words = set(stopwords.words(language))
        words = [w for w in words if w not in stop_words]
    
    return ' '.join(palabras)

#Carga del dataset
def carga_de_DS(ruta_DS):
    dataframe = pd.read_csv(ruta_DS)
    if 'Mensaje' not in dataframe.columns or 'Etiqueta' not in dataframe.columns:
        raise ValueError("Hay un error en el nombre de las columnas del DataSet")
    return dataframe

#Creacion de la matriz TF-IDF
def creacion_matriz(dataframe,**params_tdidf):
    vectorizador = TfidfVectorizer(**params_tdidf)
    X_tfidf = vectorizador.fit_transform(dataframe['texto_limpio'])
    return X_tfidf, vectorizador, vectorizador.get_feature_names_out()

#Guardar matriz en cvs
def guardar_tfidf(X_tfidf, nombres_col, etiquetas, ruta_guardado='train_tfidf.csv' ):
    dataF_tfidf = pd.DataFrame.sparse.from_spmatrix(X_tfidf, columnas = nombres_col)
    dataF_tfidf['Etiqueta'] = etiquetas.values
    dataF_tfidf.to_csv(ruta_guardado, index = False, compression='gzip')
    print("Dataset guardado en",ruta_guardado)

def guardar_vetorizador(vectorizador, dataf_limpio, ruta ):
    

def main():
    ruta_DS = 'Datasets/train.csv'
    ruta_vectorizador = 'vectorizador.pkl'
    ruta_DS_limpio = 'dataset_limpio.csv'
    ruta_DStfidf = 'tfidf.csv'
