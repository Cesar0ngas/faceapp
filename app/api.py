from flask import Flask, request, jsonify
import cv2
import numpy as np
import sys
import os

# Establece el directorio de trabajo en la carpeta de 'app'
script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

# Agrega el directorio `scripts` a la ruta de búsqueda de Python
sys.path.append(os.path.join(script_dir, '../scripts'))

from face_recognition import recognize_person  # Importa la función desde face_recognition.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Procesar la imagen recibida
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Llamar a la función de reconocimiento de persona
    result = recognize_person(img)
    print("Resultado de recognize_person:", result)

    # Asignar los valores de forma segura para evitar errores
    if isinstance(result, tuple) and len(result) == 2:
        name, probability = result
        return jsonify({"name": name, "probability": probability})
    else:
        return jsonify({"error": "No face detected or unexpected output"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
