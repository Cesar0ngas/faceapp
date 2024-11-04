import os
import requests
from flask import Flask, request, jsonify
from scripts import recognize_person, extract_face

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
    # Procesar la imagen y usar la función de reconocimiento
    # Por ejemplo, convertir a un formato adecuado y pasar a la función `recognize_person`
    image = extract_face(file)  # Aquí puedes usar una función para procesar la imagen
    name, probability = recognize_person(image)

    return jsonify({"name": name, "probability": probability})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
