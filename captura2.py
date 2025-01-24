import cv2
import subprocess
import numpy as np
import time

rtsp_url = "rtsp://admin:123456abc@192.168.1.64:554/stream"

# Comando de ffmpeg para abrir el flujo RTSP (con optimización para Jetson Nano)
command = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',
    '-i', rtsp_url,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-s', '640x360',  # Resolución reducida para mejorar el rendimiento
    '-vf', 'fps=5',  # Limitar los frames a 10 FPS
    '-an',  # Deshabilitar el audio si no lo necesitas
    '-'
]


# Abrir un proceso de ffmpeg
ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 5)

# Definir el intervalo de tiempo para sincronización (para 10 FPS)
frame_interval = 1 / 5.0  # 10 FPS

# Leer el flujo de ffmpeg
while True:
    # Registrar el tiempo antes de leer el fotograma
    start_time = time.time()

    # Leer el siguiente fotograma de ffmpeg
    raw_frame = ffmpeg_process.stdout.read(640 * 360 * 3)  # Ajusta el tamaño según la resolución
    if len(raw_frame) != 640 * 360 * 3:
        break  # Si no se recibe un frame completo, salir

    # Convertir el fotograma a un arreglo numpy
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((360, 640, 3))  # Ajusta la resolución

    # Mostrar el fotograma en una ventana de OpenCV (solo si es necesario)
    cv2.imshow("RTSP Stream", frame)

    # Calcular el tiempo restante para mantener los 10 FPS
    elapsed_time = time.time() - start_time
    remaining_time = frame_interval - elapsed_time

    # Si es necesario, esperar para mantener la sincronización
    if remaining_time > 0:
        time.sleep(remaining_time)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
ffmpeg_process.stdout.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()