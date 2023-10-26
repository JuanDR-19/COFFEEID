import cv2
import os


def segment(image):
    if image is None:
        print("No se puede cargar la imagen")
        return

    if image.shape[2] == 3:
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
        cv2.imshow("imagen combinada", combined)
        cv2.waitKey()

    elif image.shape[2] == 1:  # Grayscale image
        # Handle grayscale images as needed
        print("Input image is grayscale. Handle it accordingly.")

    else:
        print(
            "Unsupported image format. It should be either 1-channel (grayscale) or 3-channel (BGR)."
        )


def main():
    print("Identificación de granos de café maduros")
    print(
        "Se hará uso de la metodología watershed para identificar los granos de café dentro del cafeto"
    )
    target_size = (750, 800)  # Tamaño deseado para las imágenes

    # Directorio con las imágenes originales
    db = "./DB"
    valid = [".jpg", ".jpeg", ".png"]
    img_list = []

    if os.path.exists(db):
        img_list = [
            os.path.join(db, filename)
            for filename in os.listdir(db)
            if any(filename.lower().endswith(extension) for extension in valid)
        ]

    for img in img_list:
        assert (
            img is not None
        ), "El archivo de imagen no pudo ser leído, verificar que existe y que está guardado correctamente"
        image_w = cv2.imread(
            img, cv2.IMREAD_COLOR
        )  # Leer la imagen en formato BGR (3 channels)
        image_w = cv2.resize(image_w, target_size)
        segment(image_w)


if __name__ == "__main__":
    main()
