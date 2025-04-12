from fastapi import FastAPI, UploadFile, File
import io
from PIL import Image
from ocr import post_proc  # Función de OCR
from vector_extraccion import vectorize_text  # Función para vectorizar el texto
from prediccion_con_svm import hacer_predicciones, cargar_modelo  # Funciones para predicción SVM
import joblib

# Cargar el modelo SVM entrenado y el vectorizador TF-IDF
modelo_svm = cargar_modelo('svm_entrenado.pkl')
vectorizador_tfidf = joblib.load('vect_tfidf.pkl')

# Crear la aplicación FastAPI
app = FastAPI()

# Ruta para procesar la imagen con OCR y extraer texto
@app.post("/procesar-imagen/")
async def procesar_imagen(file: UploadFile = File(...)):
    try:
        # Leer la imagen en memoria
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))  # Abrir la imagen desde los bytes

        # Procesar la imagen para extraer el texto usando OCR
        texto_extraido = post_proc(image)

        return {"texto_extraido": texto_extraido}

    except Exception as e:
        return {"error": str(e)}

# Ruta para realizar la predicción usando el modelo SVM
@app.post("/predecir/")
async def predecir(texto: str):
    try:
        # Vectorizar el texto usando el archivo vector_extraccion.py
        texto_vectorizado = vectorize_text(texto, 'vect_tfidf.pkl')

        # Realizar la predicción usando el modelo SVM
        predicciones = hacer_predicciones(modelo_svm, texto_vectorizado)

        return {"prediccion": predicciones[0]}
    
    except Exception as e:
        return {"error": str(e)}
