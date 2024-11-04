import os
import requests
from flask import Flask, request, jsonify
from ..scripts.face_recognition import recognize_person  # Importa la función de reconocimiento
from ..scripts.utils.detectar import detector  # Importa el detector

app = Flask(__name__)

# Ruta local donde se guardará el modelo
LOCAL_MODEL_PATH = "models/facenet_keras.h5"
MODEL_URL = "https://storage.googleapis.com/facenet_keras/facenet_keras.h5"

def download_model():
    """Descarga el modelo de Google Cloud Storage si no existe localmente."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Descargando el modelo...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(LOCAL_MODEL_PATH, "wb") as file:
                file.write(response.content)
            print("Modelo descargado exitosamente.")
        else:
            raise Exception("Error al descargar el modelo.")

# Descargar el modelo al iniciar la API
try:
    download_model()
except Exception as e:
    print(str(e))

@app.route('/predict', methods=['POST'])
def predict():
    """Realiza la predicción a partir de una imagen enviada."""
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó la imagen.'}), 400

    file = request.files['image']
    image_bytes = file.read()  # Lee la imagen en bytes
    result = recognize_person(image_bytes)  # Asegúrate de que `recognize_person` acepte bytes

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
