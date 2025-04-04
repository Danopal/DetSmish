import cv2
import pytesseract
import numpy as np
from PIL import Image

def cargar_imagen(ruta):
    imagen = cv2.imread(ruta)
    cv2.namedWindow("Imagen cargada", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Imagen cargada", imagen)
    cv2.waitKey(0)
    return imagen

def convertir_a_grises(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Ventana autoescalable gris", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Ventana autoescalable gris", gris)  # Mostrar imagen en grises
    cv2.waitKey(0)
    return gris

def detectar_fondo(gris, umbral=127):
    """Detecta si el fondo es claro u oscuro con base en el brillo promedio"""
    brillo_promedio = np.mean(gris)
    return brillo_promedio < umbral  # True si fondo oscuro

def aplicar_umbral_fijo(gris):
    _, imagen_umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Umbral fijo aplicado", imagen_umbral) 
    cv2.waitKey(0)
    return imagen_umbral

def aplicar_umbral_adaptativo(gris, block_size=91, C=90):
    """
    Aplica un umbral adaptativo a la imagen en escala de grises.
    Se puede ajustar block_size y C para mejorar la detección.
    """
    # Invertir la imagen si el fondo es oscuro
    gris_invertido = cv2.bitwise_not(gris)
    
    # Mostrar la imagen invertida (solo para verificar si el fondo oscuro necesita inversión)
    cv2.namedWindow("Gris invertido", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Gris invertido", gris_invertido)
    cv2.waitKey(0)

    # Aplicar el umbral adaptativo
    imagen_umbral = cv2.adaptiveThreshold(gris_invertido,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,C)
    
    # Mostrar la imagen umbralizada
    cv2.namedWindow("Umbral adaptativo aplicado", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Umbral adaptativo aplicado", imagen_umbral)
    cv2.waitKey(0)
    
    return imagen_umbral

  
#Estraccion del texto de la imagen
def extraer_texto(imagen_procesada):
    return pytesseract.image_to_string(imagen_procesada, lang="spa", )

#Funcion para guardar la imagen en el txt
def guardar_texto(texto, nombre_archivo="texto_extraido.txt"): 
    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(texto)


def main():
    ruta_imagen = "imagenes-ocr/cap1.jpg" 
    imagen = cargar_imagen(ruta_imagen)
    gris = convertir_a_grises(imagen)
    fondo_oscuro = detectar_fondo(gris)

    if fondo_oscuro:
        imagen_procesada = aplicar_umbral_adaptativo(gris)
        print("Modo oscuro")

    else:
        imagen_procesada = aplicar_umbral_fijo(gris)
        print("Modo claro")

    texto = extraer_texto(imagen_procesada)
    texto_minusculas=texto.lower()
    texto_en_una_linea = texto_minusculas.replace('\n', ' ').strip()
    print(texto_en_una_linea)    
    guardar_texto(texto_en_una_linea) 



if __name__ == "__main__":
    main()
