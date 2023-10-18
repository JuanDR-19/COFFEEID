import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_pos(train_file):
    print("Cargando anotaciones y muestras positivas...")
    annotations = []
    positive_samples = []
    with open(train_file, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            if len(parts) >= 6:
                image_path = parts[0]
                x_min = int(parts[2])
                y_min = int(parts[3])
                x_max = int(parts[4])
                y_max = int(parts[5])
                annotations.append((image_path, x_min, y_min, x_max, y_max))
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    roi = image[y_min:y_max, x_min:x_max]
                    positive_samples.append(roi)
    print("Anotaciones y muestras positivas cargadas.")
    return annotations, positive_samples


def create_ps(positive_samples, target_size):
    print("Redimensionando muestras positivas...")
    samples = []
    for sample in positive_samples:
        # Redimensionar la imagen al tamaño deseado
        resized_sample = cv2.resize(
            sample, target_size
        )  # target_size debe ser una tupla (width, height)
        samples.append(resized_sample)
    print("Muestras positivas redimensionadas.")
    return samples


def train_cnn(
    positive_samples,
    negative_images_dir,
    model_filename,
    input_shape,
    num_classes,
    epochs,
    model_saving_path,
):
    print("Entrenando la red neuronal convolucional...")
    # Leer imágenes negativas
    negative_images = [
        os.path.join(negative_images_dir, img)
        for img in os.listdir(negative_images_dir)
    ]

    # Crear listas para almacenar las imágenes y etiquetas
    images = positive_samples
    labels = [1] * len(positive_samples)  # Etiqueta 1 para imágenes positivas

    # Agregar imágenes negativas a las listas
    for image_path in negative_images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Escalar la imagen al rango [0, 1] y convertirla a CV_32F
            image = cv2.normalize(
                image.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F
            )
            # Redimensionar la imagen al tamaño deseado
            resized_image = cv2.resize(
                image, input_shape[:2]
            )  # target_size debe ser una tupla (width, height)
            images.append(resized_image)
            labels.append(0)  # Etiqueta 0 para imágenes negativas

    # Convertir las listas a matrices numpy
    images = np.array(images)
    labels = np.array(labels)

    # Normalizar los valores de píxeles al rango [0, 1]
    images = images / 255.0

    # Expandir las dimensiones de las imágenes para que sean tridimensionales
    images = np.expand_dims(images, axis=-1)  # Añade una dimensión de canales

    # Crear el modelo CNN
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compilar el modelo
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Entrenar el modelo
    model.fit(images, labels, epochs=epochs)

    # Guardar el modelo
    model.save(f"{model_saving_path}/{model_filename}")
    print(f"Modelo entrenado y guardado en {model_filename}.")


def main():
    print("Identificación de granos de café maduros")
    # Directorios para imágenes positivas y negativas
    positive_images_dir = "./DB"
    negative_images_dir = "./N_DB"
    model_saving_path = "./model_training"
    # Ruta al archivo de anotaciones
    ent_file = "./DB/datos_entrenamiento.txt"

    if not os.path.isfile(ent_file):
        print(
            f"El archivo de anotaciones '{ent_file}' no se encontró. Asegúrate de que esté en la misma ubicación que tus imágenes positivas."
        )
        return

    annotations, positive_samples = load_pos(ent_file)
    target_size = (128, 128)  # Tamaño deseado para las imágenes
    num_classes = 2  # 2 clases: positiva y negativa
    epochs = 10  # Número de épocas de entrenamiento

    model_filename = "cnn_model.h5"
    if not os.path.isfile(model_filename):
        positive_samples = create_ps(positive_samples, target_size)
        input_shape = (
            target_size[0],
            target_size[1],
            1,
        )  # Tamaño de entrada de la CNN (agregando una dimensión de canal)
        train_cnn(
            positive_samples,
            negative_images_dir,
            model_filename,
            input_shape,
            num_classes,
            epochs,
            model_saving_path,
        )


if __name__ == "__main__":
    main()
