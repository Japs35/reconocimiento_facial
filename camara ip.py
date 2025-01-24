import cv2

rtsp_url = "rtsp://admin:123456abc@192.168.1.64:554/stream"
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar y mostrar la imagen
    frame_resized = cv2.resize(frame, (640, 360))
    cv2.imshow("RTSP Stream", frame_resized)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
