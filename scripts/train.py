import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

print("Cargando embeddings y etiquetas...")
# Cargar los embeddings y las etiquetas
with open('models/embeddings.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)
print("Embeddings y etiquetas cargados.")

# Codificar las etiquetas
print("Codificando etiquetas...")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
print("Etiquetas codificadas.")

# Entrenar el clasificador SVM
print("Entrenando el clasificador SVM...")
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train_encoded)
print("Clasificador entrenado.")

# Guardar el clasificador y el codificador
with open('models/svm_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Clasificador y codificador guardados correctamente.")
