import cv2
import os
import numpy as np

def train_detector(positive_images_dir, negative_images_dir, model_filename):
    # Crear una lista de imágenes positivas
    positive_images = [os.path.join(positive_images_dir, img) for img in os.listdir(positive_images_dir)]
    # Crear una lista de imágenes negativas
    negative_images = [os.path.join(negative_images_dir, img) for img in os.listdir(negative_images_dir)]
    # Crear listas para almacenar las rutas de las imágenes y etiquetas
    images = []
    labels = []
    # Agregar imágenes positivas a las listas
    for image_path in positive_images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Escalar la imagen al rango [0, 1] y convertirla a CV_32F
            image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            images.append(image)
            labels.append(1)  # Etiqueta 1 para imágenes positivas

    # Agregar imágenes negativas a las listas
    for image_path in negative_images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Escalar la imagen al rango [0, 1] y convertirla a CV_32F
            image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            images.append(image)
            labels.append(0)  # Etiqueta 0 para imágenes negativas
    # Crear un detector HOG
    # Crear un detector HOG
    hog = cv2.HOGDescriptor()
    svm = cv2.ml.SVM_create()
    # Entrenar el detector
    samples = np.array(images, dtype=np.float32)  # Cambiar el tipo de dato a CV_32F
    responses = np.array(labels, dtype=np.int32)  # Mantener las etiquetas como CV_32S
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    samples = samples.reshape(-1, np.prod(samples.shape[1:]))  # Añadir esta línea
    svm.train(samples, cv2.ml.ROW_SAMPLE, responses)
    # Guardar el modelo entrenado
    svm.save(model_filename)


def resize_images_in_directory(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    valid_extensions = ['.jpg', '.jpeg', '.png']  # Extensiones válidas de imágenes
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            # Cargar la imagen
            image = cv2.imread(input_path)
            
            if image is not None:
                # Redimensionar la imagen al tamaño deseado
                resized_image = cv2.resize(image, target_size)  # target_size debe ser una tupla (width, height)
                # Guardar la imagen redimensionada
                cv2.imwrite(output_path, resized_image)
                print(f"Imagen redimensionada y guardada: {output_path}")

def main():
    print('Identificación de granos de café maduros')
    # Directorios para imágenes positivas y negativas
    positive_images_dir = "./DB"
    negative_images_dir = "./N_DB"
    # Directorios para imágenes positivas y negativas redimensionadas
    positive_resized_dir = "./P_DB_resized"
    negative_resized_dir = "./N_DB_resized"
    # Tamaño deseado para redimensionar las imágenes
    target_size = (64, 64)
    # Redimensionar las imágenes
    resize_images_in_directory(positive_images_dir, positive_resized_dir, target_size)
    resize_images_in_directory(negative_images_dir, negative_resized_dir, target_size)
    # Nombre del archivo del modelo
    model_filename = "detector.yml"
    if not os.path.isfile(model_filename):
        train_detector(positive_resized_dir, negative_resized_dir, model_filename)
        print(f"Modelo entrenado y guardado en {model_filename}")

if __name__ == '__main__':
    main()
