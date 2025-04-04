import cv2
import pytesseract
import numpy as np

def cargar_imagen(ruta):
    return cv2.imread(ruta)

def convertir_a_grises(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def detectar_fondo(gris, umbral=127):
    """Detecta si el fondo es claro u oscuro con base en el brillo promedio"""
    brillo_promedio = np.mean(gris)
    return brillo_promedio < umbral  # True si fondo oscuro

def aplicar_umbral_fijo(gris):
    _, imagen_umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    return imagen_umbral

def aplicar_umbral_adaptativo(gris):
    # Invertimos si el fondo es oscuro para mejorar el contraste
    gris_invertido = cv2.bitwise_not(gris)
    return cv2.adaptiveThreshold(gris_invertido, 255, 
                                 cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 11, 2
                                 )

def extraer_texto(imagen_procesada):
    return pytesseract.image_to_string(imagen_procesada, lang="spa", )

def guardar_texto(texto, nombre_archivo="texto_extraido.txt"):
    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(texto)

def main():
    ruta_imagen = "imagenes-ocr/cap3.jpg"  # <- Cambiar por la tuya
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
    print("Texto extraído:\n", texto)
    
    guardar_texto(texto)
    texto_minusculas=texto.lower()
    print (texto)

if __name__ == "__main__":
    main()
