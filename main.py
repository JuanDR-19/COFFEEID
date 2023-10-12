import cv2
import os
import numpy as np

def load_pos(train_file):
    annotations = []
    with open(train_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) >= 6:
                image_path = parts[0]
                x_min = int(parts[2])
                y_min = int(parts[3])
                x_max = int(parts[4])
                y_max = int(parts[5])
                annotations.append((image_path, x_min, y_min, x_max, y_max))
    return annotations

def create_ps( annotations):
    samples = []
    for annotation in annotations:
        image_path, x_min, y_min, x_max, y_max = annotation
        image = cv2.imread(os.path.join(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            roi = image[y_min:y_max, x_min:x_max]
            samples.append(roi)
    return samples

def train(positive_samples, negative_images_dir, model_filename):
    # Leer imágenes negativas
    negative_images = [os.path.join(negative_images_dir, img) for img in os.listdir(negative_images_dir)]

    # Crear listas para almacenar las rutas de las imágenes y etiquetas
    images = positive_samples
    labels = [1] * len(positive_samples)  # Etiqueta 1 para imágenes positivas

    # Agregar imágenes negativas a las listas
    for image_path in negative_images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Escalar la imagen al rango [0, 1] y convertirla a CV_32F
            image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            images.append(image)
            labels.append(0)  # Etiqueta 0 para imágenes negativas

    # Crear un detector HOG
    hog = cv2.HOGDescriptor()
    svm = cv2.ml.SVM_create()

    # Entrenar el detector
    samples = np.array(images, dtype=np.float32)  # Cambiar el tipo de dato a CV_32F
    responses = np.array(labels, dtype=np.int32)  # Mantener las etiquetas como CV_32S
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    samples = samples.reshape(-1, np.prod(samples.shape[1:])) 
    svm.train(samples, cv2.ml.ROW_SAMPLE, responses)
    svm.save(model_filename)

def resize_images(input_dir, output_dir, target_size):
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
    # Ruta al archivo de anotaciones
    ent_file = "./DB/datos_entrenamiento.txt"

    if not os.path.isfile(ent_file):
        print(f"El archivo de anotaciones '{ent_file}' no se encontró. Asegúrate de que esté en la misma ubicación que tus imágenes positivas.")
        return

    annotations = load_pos(ent_file)
    positive_samples = create_ps(annotations)
    negative_resized_dir = "./N_DB_resized"
    target_size = (64, 64)
    resize_images(negative_images_dir, negative_resized_dir, target_size)
    model_filename = "detector.yml"
    if not os.path.isfile(model_filename):
        train(positive_samples, negative_resized_dir, model_filename)
        print(f"Modelo entrenado y guardado en {model_filename}")

if __name__ == '__main__':
    main()
