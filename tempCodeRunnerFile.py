import cv2
import os
import numpy as np


def segment(image):
    if image is None:
        print(f"No se puede cargar la imagen")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    # Calcular el histograma de color en el componente Hue
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    # Aplicar la retroproyección usando el histograma de color
    backproj = cv2.calcBackProject([hue], [0], hist, [0, 180], scale=1)
    # Aplicar la detección de bordes usando derivadas de Sobel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # Combinar la retroproyección y la detección de bordes
    combined = cv2.bitwise_and(backproj, edges)
    output_path = os.path.join("output", os.path.basename(image_path))
    cv2.imwrite(output_path, combined)
    print(f"Imagen segmentada guardada en {output_path}")


def main():
    print("Identificación de granos de café maduros")
    # Directorios para imágenes positivas y negativas
    print("Se hara uso de la metodología watershed para \n identificar los granos de cafe dentro del cafeto")
    target_size = (750, 800)  # Tamaño deseado para las imágenes
    # lectura de imagenes
    db = "./DB"  # Carpeta con las imágenes originales
    valid = [".jpg", ".jpeg", ".png"]
    img_list = []
    if os.path.exists(db):
        img_list = [
            os.path.join(db, filename)
            for filename in os.listdir(db)
            if any(filename.lower().endswith(extension) for extension in valid)
        ]

    for img in img_list:
        assert (img is not None), "el archivo de imagen no pudo ser leido, verificar que existe y que esta guardado correctamente"
        image_w = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  # lectura de las imagenes en escala de grises
        image_w = cv2.resize(image_w, target_size)
        

if __name__ == "__main__":
    main()
