# programa para la extracción de caracteres de una 
# captura de pantalla proporcionada por el usuario

'''import cv2 
import pytesseract
from PIL import Image
import matplotlib.pyplot as ploteo


captura = cv2.imread('imagenes-ocr/ima-1.jpg') # Aqui va la ruta de la imagen de la que se va a esxtraer el texto del la captura
gris = cv2.cvtColor(captura,cv2.COLOR_BG2GRAY)

umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite(umbral)'''

import pytesseract
from PIL import Image



# Cargar imagen
ruta_captura = Image.open('imagenes-ocr/ima-1.jpg')

# Extraer texto
mensaje = pytesseract.image_to_string(ruta_captura, lang='spa')

print(mensaje)