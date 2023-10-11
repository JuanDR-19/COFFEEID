import xml.etree.ElementTree as ET

xml_file = "./P_DB/training.xml"
__folder_path__ = './DB/' #direccion estatica del folder en donde se guardan las imagenes 
_extension__ = '.jpeg' #extension admitida para las imagenes de entrenamiento

tree = ET.parse(xml_file)
root = tree.getroot()
coordinates = []
labels = []
for object_elem in root.findall(".//object"):
    label = object_elem.find("name").text
    filename = object_elem.find("filename").text
    # Obtener coordenadas de la región de interés (ROI)
    bbox = object_elem.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    # Almacenar coordenadas y etiquetas
    coordinates.append((xmin, ymin, xmax, ymax))
    labels.append(label)

# Definir el nombre del archivo de salida
output_txt_file = 'datos_entrenamiento.txt'

# Abrir el archivo en modo escritura
with open(output_txt_file, 'w') as file:
    for i, (xmin, ymin, xmax, ymax) in enumerate(coordinates):
        label = labels[i]
        filename = f"{__folder_path__}{labels[i]}{filename}{_extension__}"
        # Escribir los datos en el archivo de texto
        line = f"{filename} 1 {xmin} {ymin} {xmax} {ymax}\n"
        file.write(line)

print(f'Datos de entrenamiento escritos en {output_txt_file}')

    