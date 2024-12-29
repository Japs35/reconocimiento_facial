import cv2
import os
import json
import time

# TAG de prueba

# Pedir detalles del usuario
nombre_usuario = input("Introduce tu nombre de usuario: ").strip()
nombre_completo = input("Introduce tu nombre completo: ").strip()
programa_estudios = input("Introduce tu programa de estudios: ").strip()

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
usuarios_data[nombre_usuario] = {
    'nombre_completo': nombre_completo,
    'programa_estudios': programa_estudios,
    'imagenes': []  # Lista para almacenar las imágenes del usuario
}

# Guardar los datos en el archivo JSON
with open(data_file, 'w') as f:
    json.dump(usuarios_data, f, indent=4)

# Capturar video desde la cámara
camera_ip_url = "rtsp://admin:123456abc@@192.168.1.64:554/stream"
cap = cv2.VideoCapture(camera_ip_url, cv2.CAP_FFMPEG)

#cap = cv2.VideoCapture(0)

# Verificar si la cámara se abre correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Establecer la resolución deseada (la misma usada para el reconocimiento en tiempo real)
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

count = 0
try:
    while True:
        # Leer un frame de la cámara
        ret, img = cap.read()

        # Verificar si la lectura del frame fue exitosa
        if not ret:
            print("Error: No se puede recibir el frame (¿se ha desconectado la cámara?). Saliendo...")
            break

        # Redimensionar el frame (opcional, si la cámara no soporta la resolución exacta)
        img_resized = cv2.resize(img, (desired_width, desired_height))

        # Mostrar el frame
        cv2.imshow('Capturando imágenes de entrenamiento', img_resized)

        # Capturar imagen cada vez que se presiona 'c'
        if cv2.waitKey(1) & 0xFF == ord('c'):
            img_path = os.path.join(output_dir, f'{nombre_usuario}_{count}.jpg')
            cv2.imwrite(img_path, img_resized)
            print(f"Imagen guardada en: {img_path}")

            # Guardar la ruta de la imagen en los datos del usuario
            usuarios_data[nombre_usuario]['imagenes'].append(img_path)

            # Actualizar el archivo de datos con las nuevas imágenes
            with open(data_file, 'w') as f:
                json.dump(usuarios_data, f, indent=4)

            count += 1

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Liberar el recurso de captura de video y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
