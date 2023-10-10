import cv2
import numpy as np
import os
import colors

def main():
    #proyecto vision artificial
    print('Identificación de granos de cafe maduros')
    # https://www.cenicafe.org/es/publications/arc061%2804%29315-326.pdf
    # implementación de kmeans para identificar madurez del cafe
    db = "./CoffeeID/DB"
    valid = ['.jpg', '.jpeg', '.png']
    img_list = []
    # Comprobar si el directorio existe
    if os.path.exists(db):
        # Obtener la lista de archivos en el directorio
        img_list = os.listdir(db)        
        for ph in img_list:
            if any(ph.endswith(extension) for extension in valid):
                # Construir la ruta completa al archivo de imagen
                route = os.path.join(db, ph)
                img = cv2.imread(route)
                if img is not None:
                    img_list.append(img)
    else:
        print("El directorio de imágenes no existe.")
        exit(1)
        
    grain_color = colors()
    if img is not None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Imagen con contornos", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    


if __name__ == '__main__':
    main()
