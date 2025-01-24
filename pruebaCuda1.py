import cv2
import threading

# URLs de las cámaras
rtsp_url_1 = "rtsp://admin:123456abc@192.168.1.64:554/stream"
rtsp_url_2 = "rtsp://admin:123456abc@@192.168.1.65:554/stream"  # Segunda cámara

# Función para capturar y mostrar video de una cámara
def capture_camera(rtsp_url, window_name, frames, index):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: No se puede abrir la cámara {window_name}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: No se puede leer el frame de la {window_name}")
            break

        # Usar CUDA para redimensionar la imagen
        # Convertir la imagen a formato GPU (CUDA)
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Redimensionar la imagen en la GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, (640, 360))

        # Descargar la imagen redimensionada de la GPU
        frame_resized = gpu_resized.download()

        # Guardar el frame en la lista para su uso
        frames[index] = frame_resized

    cap.release()

# Crear una lista para almacenar los frames de las cámaras
frames = [None, None]

# Crear los hilos para cada cámara
thread1 = threading.Thread(target=capture_camera, args=(rtsp_url_1, "Camara1", frames, 0))
thread2 = threading.Thread(target=capture_camera, args=(rtsp_url_2, "Camara2", frames, 1))

# Iniciar los hilos
thread1.start()
thread2.start()

# Bucle principal para mostrar las dos cámaras en la misma ventana
while True:
    if frames[0] is not None and frames[1] is not None:
        # Concatenar las dos imágenes horizontalmente
        combined_frame = cv2.hconcat([frames[0], frames[1]])

        # Mostrar la imagen combinada en una ventana
        cv2.imshow("CamarasCombinadas", combined_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Esperar que ambos hilos terminen
thread1.join()
thread2.join()

# Cerrar todas las ventanas al terminar
cv2.destroyAllWindows()
