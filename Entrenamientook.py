import cv2
import numpy as np
import os

# Verificar clasificador Haar
haarcascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(haarcascade_path):
    print(f"Error: No se encuentra {haarcascade_path}")
    exit()
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Directorio de entrenamiento
training_dir = 'imagenes_entrenamiento'
if not os.path.exists(training_dir) or not os.listdir(training_dir):
    print(f"Error: El directorio '{training_dir}' no existe o está vacío.")
    exit()

# Inicializar datos de entrenamiento
faces = []
labels = []
label_dict = {}
current_label = 0

# Recorrer los directorios de usuarios
for name_dir in os.listdir(training_dir):
    person_dir = os.path.join(training_dir, name_dir)
    if os.path.isdir(person_dir):
        # Asociar el nombre del directorio con una etiqueta numérica
        label_dict[current_label] = name_dir
        # Recorrer las imágenes dentro del directorio de cada persona
        for filename in os.listdir(person_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detectar las caras en la imagen
                faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces_rects:
                    face = gray[y:y + h, x:x + w]
                    faces.append(face)
                    labels.append(current_label)
        current_label += 1

# Verificar si se encontraron imágenes para entrenar
if len(faces) == 0 or len(labels) == 0:
    print("Error: No se encontraron datos suficientes para entrenar.")
    exit()

# Entrenar el modelo
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.save('modelo_reconocimiento.xml')
np.save('labels_dict.npy', label_dict)

print("Modelo entrenado correctamente.")
