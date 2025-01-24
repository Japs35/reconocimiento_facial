import cv2
import threading

# URLs de las cámaras
rtsp_url_1 = "rtsp://admin:123456abc@192.168.1.64:554/stream"
rtsp_url_2 = "rtsp://admin:123456abc@@192.168.1.65:554/stream"  # Segunda cámara


# Función para capturar y mostrar video de una cámara
def capture_camera(rtsp_url, window_name):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: No se puede abrir la cámara {window_name}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: No se puede leer el frame de la {window_name}")
            break

        # Redimensionar la imagen
        frame_resized = cv2.resize(frame, (640, 360))

        # Mostrar la imagen en una ventana
        cv2.imshow(window_name, frame_resized)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# Crear los hilos para cada cámara
thread1 = threading.Thread(target=capture_camera, args=(rtsp_url_1, "Camara1"))
thread2 = threading.Thread(target=capture_camera, args=(rtsp_url_2, "Camara2"))

# Iniciar los hilos
thread1.start()
thread2.start()

# Esperar que ambos hilos terminen
thread1.join()
thread2.join()

# Cerrar todas las ventanas al terminar
cv2.destroyAllWindows()
