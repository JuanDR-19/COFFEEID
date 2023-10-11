import cv2
import numpy as np
import os
import colors
from matplotlib import pyplot as plt
                    

def main():
    # Proyecto de visión artificial
    print('Identificación de granos de café maduros')
    color_white = (255, 255, 255)
    db = "./DB"
    valid = ['.jpg', '.jpeg', '.png']
    img_list = []
    grain = colors.grain() # Objeto que se utilizara para las referencias de color en RGB
    
    if os.path.exists(db):
        img_list = os.listdir(db)
        
        for ph in img_list:
            print('analizando imagen')
            if any(ph.endswith(extension) for extension in valid):
                route = os.path.join(db, ph)
                img = cv2.imread(route)
                if img is not None:
                    #sqdiff(img,template,h,w) 
                    print('Iniciando proceso de deteccion de imagenes')  
                    #TODO: buscar metodo de opencv que haga esa busqueda y seleccion mediante un archivo .txt
                            
    else:
        print("El directorio de imágenes no existe.")
        exit(1)
    
if __name__ == '__main__':
    main()
