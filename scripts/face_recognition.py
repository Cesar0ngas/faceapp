import os
import requests
import hashlib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn import MTCNN
import pickle

# URL y ruta del modelo
MODEL_URL = "https://storage.googleapis.com/facenet_keras/facenet_keras.h5"
LOCAL_MODEL_PATH = "models/facenet_keras.h5"

def verify_model_integrity(local_path, expected_md5):
    """Verifica la integridad del archivo comparando su hash MD5."""
    hash_md5 = hashlib.md5()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5

def download_model():
    """Descarga el modelo si no existe o está incompleto."""
    expected_md5 = "tu_hash_md5_aqui"  # Reemplaza con el hash MD5 real del archivo correcto
    if not os.path.exists(LOCAL_MODEL_PATH) or not verify_model_integrity(LOCAL_MODEL_PATH, expected_md5):
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        print("Descargando el modelo...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Modelo descargado exitosamente.")
            if not verify_model_integrity(LOCAL_MODEL_PATH, expected_md5):
                raise Exception("El modelo descargado está corrupto.")
        else:
            raise Exception(f"Error al descargar el modelo: {response.status_code}")
    else:
        print("El modelo ya existe localmente y está completo.")

# Descargar el modelo y cargarlo
download_model()
model = load_model(LOCAL_MODEL_PATH)

# Resto del código para el reconocimiento facial y el uso de MTCNN y clasificador
with open('models/svm_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

detector = MTCNN()

def get_embedding(face_pixels):
    """Obtiene el embedding de un rostro dado."""
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def recognize_person(frame):
    """Reconoce a la persona en una imagen capturada."""
    results = detector.detect_faces(frame)
    if results:
        for result in results:
            x1, y1, width, height = result['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            embedding = get_embedding(face)
            yhat_class = classifier.predict([embedding])[0]
            yhat_prob = classifier.predict_proba([embedding])[0]
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
