from PIL import Image
import pytesseract
import numpy as np
import cv2

# Función para cargar la imagen en memoria
def cargar_imagen(imagen: Image):
    # Convertir la imagen PIL a un arreglo NumPy
    imagen_cv = np.array(imagen)

    # Si la imagen tiene un canal alpha (transparencia), convertirla a RGB
    if imagen_cv.shape[-1] == 4:
        imagen_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_RGBA2RGB)
    
    return imagen_cv

# Función para convertir la imagen a escala de grises
def convertir_a_grises(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detecta si el fondo es oscuro o claro con base en el brillo promedio
def detectar_fondo(gris, umbral=127):
    brillo_promedio = np.mean(gris)
    return brillo_promedio < umbral  

# Aplica un umbral fijo para mejorar la calidad de la imagen
def aplicar_umbral_fijo(gris):
    _, imagen_umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    return imagen_umbral 

# Aplica un umbral adaptativo si el fondo es oscuro
def aplicar_umbral_adaptativo(gris, block_size=91, C=100):
    gris_invertido = cv2.bitwise_not(gris)
    imagen_umbral = cv2.adaptiveThreshold(gris_invertido, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    return imagen_umbral

# Extracción del texto de la imagen procesada
def extraer_texto(imagen_procesada):
    return pytesseract.image_to_string(imagen_procesada, lang="spa")

def post_proc(imagen: Image):
    try:
        imagen_cv = cargar_imagen(imagen)  # Usar imagen en memoria
        gris = convertir_a_grises(imagen_cv)
        
        fondo_oscuro = detectar_fondo(gris)
        imagen_procesada = aplicar_umbral_adaptativo(gris) if fondo_oscuro else aplicar_umbral_fijo(gris)

        texto = extraer_texto(imagen_procesada)
        texto_minusculas = texto.lower()
        texto_en_una_linea = texto_minusculas.replace('\n', ' ').strip()
        
        return texto_en_una_linea
    
    except Exception as e:
        return {"error": str(e)}
