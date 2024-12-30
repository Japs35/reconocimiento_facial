import cv2
import numpy as np
import time
import json
import pandas as pd
from datetime import datetime

# Cargar el clasificador de Haar y el modelo
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo_reconocimiento.xml')
label_dict = np.load('labels_dict.npy', allow_pickle=True).item()

# Cargar los datos del archivo JSON
data_file = 'usuarios_info.json'
with open(data_file, 'r') as f:
    usuarios_data = json.load(f)

# Cargar el logo de la institución
logo = cv2.imread('logo federal.png')  # Cambia 'logo.png' por el nombre de tu archivo de logo
logo_height, logo_width = logo.shape[:2]

# Inicializar la cámara
camera_ip_url = "rtsp://admin:123456abc@@192.168.1.64:554/stream"
cap = cv2.VideoCapture(camera_ip_url, cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: no se puede abrir la cámara")
    exit()

# Configurar resolución
desired_width = 1280  # Cambiar según la resolución nativa de la cámara
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Establecer un umbral de confianza
confidence_threshold = 80  # Ajusta este valor para incrementar la confianza

# Variables para controlar la cantidad de frames procesados
frame_count = 0
frame_skip = 3  # Procesar un frame cada 3 frames

# Variables para el mensaje de bienvenida
last_detected_name = None
detection_start_time = None  # Tiempo cuando se detectó la cara por primera vez
detection_timeout = 3  # Tiempo en segundos para mostrar el mensaje (3 segundos)
detection_lost_time = 5  # Tiempo en segundos antes de que desaparezca el mensaje si no se detecta

# Variables para controlar el registro de los usuarios
registered_users = set()  # Conjunto para guardar nombres de usuarios ya registrados
message_display_time = None  # El tiempo en que se debe mostrar el mensaje
message_duration = 3  # Duración en segundos del mensaje

# Crear DataFrame vacío para almacenar registros
df = pd.DataFrame(columns=['Timestamp', 'Nombre', 'Programa Estudio'])

# Generar el primer archivo Excel con la fecha y hora actuales
fecha_actual = datetime.now().strftime("%d.%m.%Y.%H.%M")
archivo_excel = f"Registro{fecha_actual}.xlsx"
df.to_excel(archivo_excel, index=False)
print(f"Primer archivo Excel generado: {archivo_excel}")

# Variable para controlar el tiempo de 10 minutos
last_excel_time = time.time()

# Variable para manejar el conteo de los 3 segundos continuos
face_detected_time = 0
detected = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: no se puede recibir el frame")
            continue

        # Solo procesamos un frame cada 'frame_skip' frames
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Saltamos este frame

        # Redimensionar al tamaño deseado para procesamiento y visualización
        output_width = 640
        output_height = 480
        frame_resized = cv2.resize(frame, (output_width, output_height))

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_name = None  # Variable para almacenar el nombre detectado en este frame

        for (x, y, w, h) in faces_rects:
            face = gray[y:y + h, x:x + w]
            label, confidence = face_recognizer.predict(face)

            # Si la confianza es menor que el umbral, se considera como un usuario conocido
            if confidence < confidence_threshold:
                detected_name = label_dict[label]
                detected = True
            else:
                detected_name = "Usuario Externo"  # Si la confianza es alta, es un usuario desconocido

            # Dibujar el rectángulo y el nombre
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_resized, detected_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Si se detecta un usuario válido y no se ha registrado antes
        if detected and detected_name != "Usuario Externo" and detected_name not in registered_users:
            if last_detected_name != detected_name or last_detected_name is None:
                # Nueva detección, iniciar temporizador
                detection_start_time = time.time()
                last_detected_name = detected_name
                face_detected_time = time.time()  # Iniciar conteo desde la detección

            # Si la persona se mantiene frente a la cámara durante 3 segundos continuos
            elif time.time() - face_detected_time >= detection_timeout:
                user_info = usuarios_data.get(detected_name, {})
                nombre_completo = user_info.get("nombre_completo", "Nombre Desconocido")
                programa_estudios = user_info.get("programa_estudios", "Programa Desconocido")

                # Agregar el registro al DataFrame
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df = pd.concat([df, pd.DataFrame(
                    [{'Timestamp': timestamp, 'Nombre': nombre_completo, 'Programa Estudio': programa_estudios}])],
                               ignore_index=True)

                # Marcar este usuario como registrado
                registered_users.add(detected_name)

                # Guardar el DataFrame en un archivo Excel
                fecha_actual = datetime.now().strftime("%d.%m.%Y.%H.%M")
                archivo_excel = f"Registro{fecha_actual}.xlsx"
                df.to_excel(archivo_excel, index=False)
                print(f"Registro guardado en: {archivo_excel}")

                # Configurar el tiempo para mostrar el mensaje de bienvenida
                message_display_time = time.time()

        # Mostrar el mensaje de bienvenida durante 3 segundos
        if message_display_time and time.time() - message_display_time <= message_duration:
            # Ajustar el tamaño del rectángulo negro según el texto
            message_lines = [
                'Bienvenido',
                f'Nombre: {nombre_completo}',
                f'Programa: {programa_estudios}'
            ]
            text_height = 0
            for line in message_lines:
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_height += h + 10  # Añadimos un pequeño margen entre las líneas

            # Coordenadas para el fondo negro ajustado al texto
            rect_top_left = (50, frame_resized.shape[0] - text_height - 10)
            rect_bottom_right = (frame_resized.shape[1] - 50, frame_resized.shape[0] - 10)

            # Dibujar el rectángulo negro
            cv2.rectangle(frame_resized, rect_top_left, rect_bottom_right, (0, 0, 0), -1)

            # Mostrar el mensaje de bienvenida con un tamaño de texto más pequeño
            y_offset = frame_resized.shape[0] - text_height - 10
            for line in message_lines:
                cv2.putText(frame_resized, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y_offset += h + 10  # Mover el siguiente texto abajo

        # Generar nuevo archivo Excel cada 10 minutos
        if time.time() - last_excel_time >= 600:  # 10 minutos en segundos
            last_excel_time = time.time()
            fecha_actual = datetime.now().strftime("%d.%m.%Y.%H.%M")
            archivo_excel = f"Registro{fecha_actual}.xlsx"
            df.to_excel(archivo_excel, index=False)
            print(f"Nuevo archivo Excel generado: {archivo_excel}")

        # Mostrar el frame
        cv2.imshow('Reconocimiento Facial en Tiempo Real', frame_resized)

        # Romper el ciclo si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
