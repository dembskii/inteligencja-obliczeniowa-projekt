from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
            # Tutaj wywołasz swoją sieć neuronową, na razie dummy:
            predicted_species = "Nieznany gatunek (dummy)"
            return render_template('result.html', filename=filename, species=predicted_species)
        else:
            return render_template('index.html', error='Niedozwolony format pliku!')
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(port=3000, debug=True)