from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ścieżka do modelu (dostosuj ścieżkę jeśli model jest w innym miejscu)
MODEL_PATH = '../neural_network_1/chess_pieces_model.h5'

# Załaduj model
model = tf.keras.models.load_model(MODEL_PATH)

# Dodaj ten kod, aby wyświetlić szczegółowe informacje o modelu
print("Model summary:")
model.summary()

# Nazwy klas figur szachowych - PRZENIESIONE TUTAJ (przed funkcją debug_model_classes)
class_names = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']

# Funkcja debugująca do weryfikacji kolejności klas
def debug_model_classes():
    # Ta funkcja wymaga przygotowania przykładowego zdjęcia każdej figury
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_images')
    if not os.path.exists(debug_dir):
        print("Folder debug_images nie istnieje, pomijanie weryfikacji klas")
        return
    
    print("\n=== WERYFIKACJA KOLEJNOŚCI KLAS W MODELU ===")
    # Załóżmy, że mamy przykładowe zdjęcia nazwane odpowiednio: bishop.jpg, king.jpg, itd.
    for expected_class in class_names:
        img_path = os.path.join(debug_dir, f"{expected_class}.jpg")
        if not os.path.exists(img_path):
            print(f"Brak pliku {img_path}, pomijanie")
            continue
            
        img = preprocess_image(img_path)
        pred = model.predict(img, verbose=0)
        pred_class_idx = np.argmax(pred[0])
        confidence = pred[0][pred_class_idx] * 100
        
        # Porównujemy indeks przewidzianej klasy z jej pozycją w class_names
        expected_idx = class_names.index(expected_class)
        print(f"Dla figury {expected_class}: model przewidział klasę {pred_class_idx} z pewnością {confidence:.2f}%")
        print(f"  Powinno być: {expected_idx}")
        
        # Pełny wektor przewidywań 
        for i, prob in enumerate(pred[0]):
            print(f"  Klasa {i}: {prob*100:.2f}%")
    
    print("=== KONIEC WERYFIKACJI ===\n")

def preprocess_image(filepath):
    """Przygotowanie obrazu do predykcji przez model"""
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalizacja

# Uruchom weryfikację przy starcie (przeniesione po definicji preprocess_image)
debug_model_classes()

# Mapowanie angielskich nazw na polskie
pieces_translation = {
    'bishop': 'Goniec',
    'king': 'Król',
    'knight': 'Skoczek',
    'pawn': 'Pionek',
    'queen': 'Królowa',
    'rook': 'Wieża'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Brak pliku!')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Nie wybrano pliku!')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Rozpoznawanie figury szachowej
            try:
                processed_img = preprocess_image(filepath)
                predictions = model.predict(processed_img)
                predicted_class_index = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_index]) * 100
                
                # Nazwa figury po angielsku
                predicted_class = class_names[predicted_class_index]
                
                # Nazwa figury po polsku
                figure_name = pieces_translation.get(predicted_class, predicted_class)
                
                return render_template('result.html', 
                                      filename='uploads/' + filename, 
                                      species=figure_name,
                                      confidence=confidence)
            except Exception as e:
                return render_template('index.html', error=f'Błąd w przetwarzaniu obrazu: {str(e)}')
        else:
            return render_template('index.html', error='Niedozwolony format pliku!')
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)