import os
import requests
import hashlib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN

# Ruta del modelo en Google Cloud Storage y donde se guardará localmente
MODEL_URL = "https://storage.googleapis.com/facenet_keras/facenet_keras.h5"
LOCAL_MODEL_PATH = "models/facenet_keras.h5"
EXPECTED_MD5 = "d4169b76ead0a7a58c5ba7ca4c0b505b"  # Hash MD5 del modelo esperado

# Verificar la integridad del archivo descargado
def verify_md5(file_path, expected_md5):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest() == expected_md5

# Descargar el modelo si no existe o si la verificación MD5 falla
def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Descargando el modelo...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(LOCAL_MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("Modelo descargado exitosamente.")
        else:
            raise Exception("Error al descargar el modelo.")

    # Verificar la integridad del archivo descargado
    if not verify_md5(LOCAL_MODEL_PATH, EXPECTED_MD5):
        os.remove(LOCAL_MODEL_PATH)
        raise Exception("El modelo descargado está corrupto o no es válido.")

# Llama a download_model para asegurar que el modelo esté descargado y verificado
download_model()

# Cargar el modelo FaceNet
try:
    model = load_model(LOCAL_MODEL_PATH)
    print("Modelo FaceNet cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Cargar el clasificador SVM y el codificador de etiquetas
try:
    with open("models/svm_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"Error al cargar el clasificador o el codificador: {e}")

# Inicializar el detector de rostros MTCNN
detector = MTCNN()

# Función para obtener el embedding de un rostro
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype("float32")
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Función para reconocer a la persona en un frame de video o imagen
def recognize_person(frame):
    results = detector.detect_faces(frame)
    if results:  # Si hay al menos un rostro detectado
        for result in results:
            x1, y1, width, height = result["box"]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            
            # Obtener el embedding y realizar la predicción
            embedding = get_embedding(face)
            yhat_class = classifier.predict([embedding])[0]
            yhat_prob = classifier.predict_proba([embedding])[0]

            # Intentar asignar nombre y probabilidad
            try:
                class_probability = yhat_prob[yhat_class] * 100
                predicted_name = encoder.inverse_transform([yhat_class])[0]
                return predicted_name, class_probability
            except (IndexError, ValueError) as e:
                print(f"Error en la predicción: {e}")
                return "Desconocido", 0.0
    else:
        print("No se detectó ningún rostro")
        return "Desconocido", 0.0
