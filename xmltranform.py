import xml.etree.ElementTree as ET

xml_file = "./P_DB/training.xml"
__folder_path__ = './DB/'  # Dirección estática del folder donde se guardan las imágenes
__extension__ = '.jpeg'  # Extensión admitida para las imágenes de entrenamiento
__folder__ = 'P_DB'

tree = ET.parse(xml_file)
root = tree.getroot()
coordinates = []
labels = []

for object_elem in root.findall(".//object"):
    label = object_elem.find("name").text
    # Obtener coordenadas de la región de interés (ROI)
    bbox = object_elem.find("bndbox")
    xmin = int(float(bbox.find("xmin").text))
    ymin = int(float(bbox.find("ymin").text))
    xmax = int(float(bbox.find("xmax").text))
    ymax = int(float(bbox.find("ymax").text))
    
    # Almacenar coordenadas y etiquetas
    coordinates.append((xmin, ymin, xmax, ymax))
    labels.append(label)

# Definir el nombre del archivo de salida
output_txt_file = f'{__folder__}/datos_entrenamiento.txt'

# Abrir el archivo en modo escritura
with open(output_txt_file, 'w') as file:
    for i, (xmin, ymin, xmax, ymax) in enumerate(coordinates):
        label = labels[i]
        # Escribir los datos en el archivo de texto
        line = f"1 {xmin} {ymin} {xmax} {ymax}\n"
        file.write(line)

print(f'Datos de entrenamiento escritos en {output_txt_file}')
