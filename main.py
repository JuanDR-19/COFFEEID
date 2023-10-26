import cv2
import os
import numpy as np

def main():
    print("Identificación de granos de café maduros")
    print("Se hará uso de la metodología watershed para identificar los granos de café dentro del cafeto")
    target_size = (750, 800)  # Tamaño deseado para las imágenes
    # Directorio con las imágenes originales
    db = "./DB"
    valid = [".jpg", ".jpeg", ".png"]
    img_list = []
    redBajo1 = np.array([0, 100, 20], np.uint8)
    redAlto1 = np.array([8, 255, 255], np.uint8)
    redBajo2 = np.array([175, 100, 20], np.uint8)
    redAlto2 = np.array([179, 255, 255], np.uint8)

    if os.path.exists(db):
        img_list = [
            os.path.join(db, filename)
            for filename in os.listdir(db)
            if any(filename.lower().endswith(extension) for extension in valid)
        ]

    for img in img_list:
        assert (img is not None), "El archivo de imagen no pudo ser leído, verificar que existe y que está guardado correctamente"
        image_w = cv2.imread(img, cv2.IMREAD_COLOR)  # Leer la imagen en formato BGR (3 channels)
        image_w = cv2.resize(image_w, target_size) # cambiar el tamaño de las imagnes para dar estandar
        image_w = cv2.cvtColor(image_w, cv2.COLOR_BGR2HSV)  # cambiar el color a escala hsv
        maskRed1 = cv2.inRange(image_w, redBajo1, redAlto1) # aplicar las mascaras
        maskRed2 = cv2.inRange(image_w, redBajo2, redAlto2) # aplicar las segundas mascaras
        maskRed = cv2.add(maskRed1, maskRed2) #combinar mascaras
        maskRedvis = cv2.bitwise_and(image_w, image_w, mask= maskRed) #binarizar las imagenes
        cv2.imshow('frame', maskRedvis) #mostrar la imagen binarizada
        cv2.waitKey()


if __name__ == "__main__":
    main()
