import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import pickle

# Obtener la ruta absoluta para el modelo FaceNet y los archivos del clasificador y codificador
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Ruta del directorio raíz del proyecto
model_path = os.path.join(base_dir, 'models', 'facenet_keras.h5')
classifier_path = os.path.join(base_dir, 'models', 'svm_classifier.pkl')
encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')

# Verificar que todos los archivos existen
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo de modelo no se encontró en la ruta: {model_path}")
if not os.path.exists(classifier_path):
    raise FileNotFoundError(f"El archivo del clasificador no se encontró en la ruta: {classifier_path}")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"El archivo del codificador no se encontró en la ruta: {encoder_path}")

# Cargar el modelo FaceNet, el clasificador y el codificador
model = load_model(model_path)
print("Modelo FaceNet cargado exitosamente.")
with open(classifier_path, 'rb') as f:
    classifier = pickle.load(f)
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Inicializar el detector de rostros MTCNN
detector = MTCNN()

# Función para obtener el embedding de un rostro
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
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
            x1, y1, width, height = result['box']
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
