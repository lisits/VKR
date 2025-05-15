import pandas as pd
from sqlalchemy import create_engine
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Конфигурация
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'VKR',
    'user': 'postgres',
    'password': 'postgres'
}

MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_WORDS = 5000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 100

# 1. Подключение к БД и загрузка данных
engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

query = "SELECT title, description, original_estimate, role_bool FROM yolva_data_short"
data = pd.read_sql(query, engine)

# 2. Предобработка данных
# Обработка текстовых полей
data['text'] = data['title'].fillna('') + ' ' + data['description'].fillna('')
data['text'] = data['text'].astype(str)

# Нормализация времени
scaler = StandardScaler()
y_time = scaler.fit_transform(data['original_estimate'].values.reshape(-1, 1))
y_role = data['role_bool'].values

# 3. Подготовка текстовых данных
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
X_text = pad_sequences(sequences, maxlen=MAX_LEN)

# 4. Разделение данных
X_train, X_test, y_time_train, y_time_test, y_role_train, y_role_test = train_test_split(
    X_text, y_time, y_role, test_size=0.2, random_state=42
)


# 5. Создание модели
def create_model():
    input_layer = Input(shape=(MAX_LEN,))

    # Текстовая обработка
    x = Embedding(MAX_WORDS, 128)(input_layer)
    x = LSTM(64, dropout=0.3, return_sequences=True)(x)
    x = LSTM(32, dropout=0.2)(x)

    # Общие слои
    shared = Dense(64, activation='relu')(x)
    shared = Dropout(0.4)(shared)

    # Ветвь для времени
    time_branch = Dense(32, activation='relu')(shared)
    time_output = Dense(1, name='time_output')(time_branch)

    # Ветвь для роли
    role_branch = Dense(16, activation='relu')(shared)
    role_output = Dense(1, activation='sigmoid', name='role_output')(role_branch)

    model = Model(
        inputs=input_layer,
        outputs=[time_output, role_output]
    )

    model.compile(
        optimizer='adam',
        loss={
            'time_output': 'mse',
            'role_output': 'binary_crossentropy'
        },
        loss_weights={'time_output': 0.7, 'role_output': 0.3},
        metrics={
            'time_output': ['mae'],
            'role_output': ['accuracy']
        }
    )

    return model


def save_model_artifacts(model, tokenizer, scaler, model_dir=MODEL_DIR):
    """Сохраняет модель и вспомогательные артефакты"""
    # Создаем подпапку с текущей датой и временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(model_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Сохраняем модель
    model_path = os.path.join(save_dir, 'model.keras')
    model.save(model_path)

    # Сохраняем токенизатор
    tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Сохраняем скейлер
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Сохраняем архитектуру модели в виде картинки
    plot_path = os.path.join(save_dir, 'model_architecture.png')
    try:
        plot_model(model, to_file=plot_path, show_shapes=True)
    except:
        pass

    print(f"Модель и артефакты сохранены в {save_dir}")
    return save_dir


def load_model_artifacts(model_dir):
    """Загружает модель и вспомогательные артефакты"""
    # Загружаем модель
    model_path = os.path.join(model_dir, 'model.keras')
    model = load_model(model_path)

    # Загружаем токенизатор
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Загружаем скейлер
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Модель и артефакты загружены из {model_dir}")
    return model, tokenizer, scaler


# Создаем или загружаем модель
TRAIN_NEW_MODEL = True  # Установите False для загрузки существующей модели

if TRAIN_NEW_MODEL:
    model = create_model()

    # 6. Обучение
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        {
            'time_output': y_time_train,
            'role_output': y_role_train
        },
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Сохраняем модель и артефакты
    saved_dir = save_model_artifacts(model, tokenizer, scaler)
else:
    # Загружаем последнюю сохраненную модель
    model_dirs = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    if not model_dirs:
        raise ValueError("Нет сохраненных моделей для загрузки")

    latest_dir = sorted(model_dirs)[-1]
    model, tokenizer, scaler = load_model_artifacts(os.path.join(MODEL_DIR, latest_dir))

# 7. Оценка модели
print("\nОценка модели на тестовых данных:")
test_loss = model.evaluate(
    X_test,
    {
        'time_output': y_time_test,
        'role_output': y_role_test
    },
    verbose=0
)

print(f"Общий лосс: {test_loss[0]:.4f}")
print(f"Точность выбора роли: {test_loss[4] * 100:.1f}%")

# Предсказания модели
y_pred_time, y_pred_role = model.predict(X_test, verbose=0)

# Обратное преобразование времени
y_pred_time_inverse = scaler.inverse_transform(y_pred_time)
y_time_test_inverse = scaler.inverse_transform(y_time_test)

# Метрики регрессии (время)
mae = mean_absolute_error(y_time_test_inverse, y_pred_time_inverse)
rmse = np.sqrt(mean_squared_error(y_time_test_inverse, y_pred_time_inverse))
r2 = r2_score(y_time_test_inverse, y_pred_time_inverse)

print("\nМетрики для предсказания времени:")
print(f"MAE: {mae:.2f} часов")
print(f"RMSE: {rmse:.2f} часов")
print(f"R²: {r2:.3f}")

# Метрики классификации (роль)
y_pred_role_binary = (y_pred_role > 0.5).astype(int)

accuracy = accuracy_score(y_role_test, y_pred_role_binary)
precision = precision_score(y_role_test, y_pred_role_binary)
recall = recall_score(y_role_test, y_pred_role_binary)
f1 = f1_score(y_role_test, y_pred_role_binary)

print("\nМетрики для предсказания роли:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# 8. Функция для предсказания
class TaskPredictor:
    def __init__(self, model, tokenizer, scaler):
        self.model = model
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.max_len = MAX_LEN

    def predict_task(self, title, description):
        """Предсказывает время и рекомендует исполнителя"""
        text = title + ' ' + description

        # Токенизация
        seq = self.tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(seq, maxlen=self.max_len)

        # Предсказание
        pred_time, pred_role = self.model.predict(padded_seq, verbose=0)

        # Обработка результатов
        hours = self.scaler.inverse_transform(pred_time)[0][0]
        role_prob = pred_role[0][0]
        recommendation = 'разработчик' if role_prob > 0.5 else 'аналитик'

        return {
            'title': title,
            'description': description,
            'estimated_hours': round(hours, 1),
            'recommended_role': recommendation,
            'role_confidence': round(float(role_prob if recommendation == 'разработчик' else 1 - role_prob), 3)
        }

# Создаем экземпляр предсказателя
predictor = TaskPredictor(model, tokenizer, scaler)

# 9. Пример использования
sample_task = {
    'title': "Разработать REST API для интеграции с платежной системой",
    'description': "Необходимо Разработать REST API для интеграции с платежной системой SberPay"
}

result = predictor.predict_task(**sample_task)
print("\nРезультат предсказания:")
print(f"Задача: {result['title']}")
print(f"Описание: {result['description']}")
print(f"Прогноз времени: {result['estimated_hours']} часов")
print(f"Рекомендуемый исполнитель: {result['recommended_role']} (уверенность: {result['role_confidence'] * 100:.1f}%)")