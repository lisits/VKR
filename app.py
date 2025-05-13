from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from werkzeug.utils import secure_filename
from uuid import uuid4

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    file = request.files.get('file')
    if not file:
        return jsonify({'success': False, 'error': 'Файл не найден.'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка чтения файла: {e}'})

    if 'Задача' not in df.columns or 'Описание' not in df.columns:
        return jsonify({'success': False, 'error': 'Файл должен содержать колонки "Задача" и "Описание"'})

    # Пример расчёта: просто ставим заглушки
    df['Роль'] = 'Разработчик'
    df['Часы'] = 8

    result_filename = f'result_{uuid4().hex}.xlsx'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    df[['Задача', 'Описание', 'Роль', 'Часы']].to_excel(result_path, index=False)

    return jsonify({'success': True, 'result_file': result_filename})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data.get('title', '')
    description = data.get('description', '')

    if not title or not description:
        return jsonify({'success': False, 'error': 'Поля не заполнены.'})

    # Пример: роль и оценка — заглушки
    return jsonify({
        'success': True,
        'task': {
            'role': 'Аналитик',
            'hours': 6
        }
    })

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

