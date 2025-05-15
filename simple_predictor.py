import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class PredictorSimple:
    def __init__(self, model_dir='saved_models'):
        # Находим последнюю сохранённую модель
        model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if not model_dirs:
            raise ValueError("Нет сохраненных моделей в папке saved_models")

        latest_dir = sorted(model_dirs)[-1]
        self.model_path = os.path.join(model_dir, latest_dir, 'model.keras')
        self.tokenizer_path = os.path.join(model_dir, latest_dir, 'tokenizer.pkl')
        self.scaler_path = os.path.join(model_dir, latest_dir, 'scaler.pkl')

        # Загрузка модели и артефактов
        self.model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.max_len = 100  # Должен совпадать с MAX_LEN из обучения

    def predict_task(self, title, description):
        text = title + ' ' + description
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len)

        # Предсказание
        pred_time, pred_role = self.model.predict(padded, verbose=0)
        hours = self.scaler.inverse_transform(pred_time)[0][0]
        role_prob = pred_role[0][0]
        role = 'разработчик' if role_prob > 0.5 else 'аналитик'
        confidence = round(float(role_prob if role == 'разработчик' else 1 - role_prob), 3)

        return {
            'title': title,
            'description': description,
            'estimated_hours': round(hours, 1),
            'recommended_role': role,
            'role_confidence': confidence
        }