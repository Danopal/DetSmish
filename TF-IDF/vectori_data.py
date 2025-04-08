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
        palabras = [w for w in palabras if w not in stop_words]
    
    # Lematización 
    if lemmatize:
        palabras = [w if w not in string.punctuation else "" for w in palabras] 

    
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
    X_tfidf = vectorizador.fit_transform(dataframe['preprocesamiento'])
    return X_tfidf, vectorizador, vectorizador.get_feature_names_out()

#Guardar matriz en cvs
def guardar_tfidf(X_tfidf,feature_names,etiquetas, ruta_guardado='train_tfidf.csv' ):
    dataF_tfidf = pd.DataFrame.sparse.from_spmatrix(X_tfidf, columns = feature_names)
    dataF_tfidf['Etiqueta'] = etiquetas.values
    dataF_tfidf.to_csv(ruta_guardado, index = False) #     dataF_tfidf.to_csv(ruta_guardado, index = False,
    print(f"Dataset guardado en {ruta_guardado} " )

def guardar_vetorizador(vectorizador, dataf_limpio, ruta_vectorizador = 'vectorizador_tfidf.pkl', ruta_DS_limpio = 'DS_limpio.cvs'):
    joblib.dump(vectorizador, ruta_vectorizador)
    print(f"vectorizador guardado en {ruta_vectorizador}")
    dataf_limpio.to_csv(ruta_DS_limpio, index= False)
    print(f"Datos guardados en {ruta_DS_limpio}")


def main():
    RUTA_DS = 'Datasets/train.csv'
    RUTA_VECTORIZADOR = 'vectorizador.pkl'
    RUTA_DS_LIMPIO = 'dataset_limpio.csv'
    RUTA_DS_TFIDF = 'tfidf.csv'

    print("cargando los datos")
    dataframe = carga_de_DS(RUTA_DS)

    print("Limpiando el texto...")
    dataframe['preprocesamiento'] = dataframe['Mensaje'].apply(lambda x: preprocesamiento(x,remove_punct=True, remove_numbers = True, remove_stopwords = True, lemmatize = True))

    print("Vectorizando el DataSet")
    X_tfidf, vectorizador, feature_names = creacion_matriz(dataframe, max_features=15000, ngram_range=(1, 2))

    guardar_tfidf(X_tfidf ,feature_names, dataframe['Etiqueta'],RUTA_DS_TFIDF)

    guardar_vetorizador(vectorizador, dataframe, RUTA_VECTORIZADOR, RUTA_DS_LIMPIO)

    print("Proceso realizado correctamente")

if __name__ == "__main__":
    main()    