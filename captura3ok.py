import cv2
import os
import json

# Pedir detalles del usuario
nombre_usuario = input("Introduce tu nombre de usuario: ").strip()

# Crear el directorio para guardar las imágenes si no existe
output_dir = os.path.join('imagenes_entrenamiento', nombre_usuario)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ruta del archivo donde guardaremos la información del usuario
data_file = 'usuarios_info.json'

# Si el archivo de datos ya existe, cargamos la información
if os.path.exists(data_file):
    with open(data_file, 'r') as f:
        usuarios_data = json.load(f)
else:
    usuarios_data = {}

# Guardamos la información del nuevo usuario en el archivo
if nombre_usuario not in usuarios_data:
    usuarios_data[nombre_usuario] = {
        'nombre_completo': nombre_usuario,
        'imagenes': []  # Lista para almacenar las imágenes del usuario
    }

# Guardar los datos en el archivo JSON
with open(data_file, 'w') as f:
    json.dump(usuarios_data, f, indent=4)

# Capturar video desde la cámara
rtsp_url = "rtsp://admin:123456abc@192.168.1.64:554/stream"
cap = cv2.VideoCapture(rtsp_url)

# Contador de imágenes
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar y mostrar la imagen
    frame_resized = cv2.resize(frame, (640, 360))
    cv2.imshow("RTSP Stream", frame_resized)

    # Detectar si se presionó 'c' para capturar la imagen
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Definir la ruta para guardar la imagen
        img_path = os.path.join(output_dir, f'{nombre_usuario}_{count}.jpg')

        # Guardar la imagen
        cv2.imwrite(img_path, frame_resized)
        print(f"Imagen guardada en: {img_path}")

        # Guardar la ruta de la imagen en los datos del usuario
        usuarios_data[nombre_usuario]['imagenes'].append(img_path)

        # Actualizar el archivo de datos con las nuevas imágenes
        with open(data_file, 'w') as f:
            json.dump(usuarios_data, f, indent=4)

        # Incrementar el contador de imágenes
        count += 1

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
