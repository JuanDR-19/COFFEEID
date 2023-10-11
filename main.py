import cv2
import numpy as np
import os
import colors
from matplotlib import pyplot as plt


def contorno(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,0,255), 2)
    cv2.imshow("Imagen con contornos", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def sqdiff(img, template, h, w):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_grey, template, cv2.TM_SQDIFF)
    plt.imshow(res, cmap='gray')
    plt.title('Resultado de la coincidencia')
    plt.show()
    identify(h,w,res,img_grey)
    
    

def identify(h, w, res, img_grey):
    # Señalar los granos seleccionados
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res) #encontrar los valores mínimo y máximo y sus ubicaciones en la matriz res
    # min_loc se utiliza para identificar la ubicación del punto donde se encontró la mejor coincidencia.
    top_left = min_loc #top_left es la esquina superior izquierda del rectángulo que se utilizará para señalar la ubicación del grano de café identificado.
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_with_rectangle = img_grey.copy()
    cv2.rectangle(img_with_rectangle, top_left, bottom_right, 255, 2)
    cv2.imshow('Imagen con Rectángulo', img_with_rectangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

                    

def main():
    # Proyecto de visión artificial
    print('Identificación de granos de café maduros')
    color_white = (255, 255, 255)
    template = cv2.imread('./knowledge/template.jpeg', cv2.IMREAD_GRAYSCALE)  # Cargar la plantilla como imagen en escala de grises
    h, w = template.shape[:2]
    db = "./DB"
    valid = ['.jpg', '.jpeg', '.png']
    img_list = []
    grain = colors.grain()
    
    if os.path.exists(db):
        img_list = os.listdir(db)
        
        for ph in img_list:
            print('analizando imagen')
            if any(ph.endswith(extension) for extension in valid):
                route = os.path.join(db, ph)
                img = cv2.imread(route)
                if img is not None:
                    sqdiff(img,template,h,w)   
                    
    else:
        print("El directorio de imágenes no existe.")
        exit(1)
    
if __name__ == '__main__':
    main()
