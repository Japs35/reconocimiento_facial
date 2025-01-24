import cv2
import threading
import numpy as np

# URLs de las cámaras IP
camera_1_url = "rtsp://admin:123456abc@192.168.1.64:554/stream"
camera_2_url = "rtsp://admin:123456abc@@192.168.1.65:554/stream"

# Parámetros de resolución
desired_width = 640
desired_height = 480

# Cargar el clasificador Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo entrenado y el diccionario de etiquetas
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo_reconocimiento.xml')
label_dict = np.load('labels_dict.npy', allow_pickle=True).item()

# Evento para controlar la ejecución de los hilos
stop_event = threading.Event()

# Función para procesar una cámara
def process_camera(camera_url, camera_name):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)

    # Verificar si la cámara se abre correctamente
    if not cap.isOpened():
        print(f"Error: No se puede abrir la cámara {camera_name}")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: No se puede recibir el frame de la cámara {camera_name}. Saliendo...")
            break

        # Redimensionar el frame para ajustarse a la resolución deseada
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar rectángulos y mostrar nombres
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(roi_gray)
            if confidence < 75:  # Umbral de confianza (ajustable)
                name = label_dict.get(label, "Desconocido")
            else:
                name = "Desconocido"

            # Dibujar el rectángulo celeste
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 50), 2)

            # Mostrar el nombre color celeste
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 50), 2)

        # Mostrar el video
        cv2.imshow(f'Cámara {camera_name}', frame)

        # Salir al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# Crear hilos para procesar ambas cámaras en paralelo
thread_1 = threading.Thread(target=process_camera, args=(camera_1_url, "1"))
thread_2 = threading.Thread(target=process_camera, args=(camera_2_url, "2"))

# Iniciar los hilos
thread_1.start()
thread_2.start()

# Monitorear los hilos y cerrar al presionar 'q'
try:
    thread_1.join()
    thread_2.join()
finally:
    # Asegurar el cierre de todas las ventanas
    stop_event.set()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")
