import cv2
import numpy as np
import os
# import colors

def main():
    # Proyecto de visión artificial
    print('Identificación de granos de café maduros')
    db = "./DB"
    valid = ['.jpg', '.jpeg', '.png']
    img_list = []
    
    if os.path.exists(db):
        img_list = os.listdir(db)
        
        for ph in img_list:
            if any(ph.endswith(extension) for extension in valid):
                route = os.path.join(db, ph)
                img = cv2.imread(route)
                
                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
                    edges = cv2.Canny(blurred_img, 50, 150)
                    kernel = np.ones((5, 5), np.uint8)
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                    
                    # Guardar la imagen con los contornos
                    output_path = os.path.join(db, "contours_" + ph)
                    cv2.imwrite(output_path, img)
                    
                    cv2.imshow("Imagen con contornos", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:
        print("El directorio de imágenes no existe.")
        exit(1)
    
if __name__ == '__main__':
    main()
