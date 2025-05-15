import pandas as pd
from sqlalchemy import create_engine
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from transformers import BertTokenizer, TFBertModel
import numpy as np

# Конфигурация
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'VKR',
    'user': 'postgres',
    'password': 'postgres'
}

MAX_LEN = 100
BATCH_SIZE = 16
EPOCHS = 10

# 1. Подключение к БД и загрузка данных
engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

query = "SELECT title, description, original_estimate, role_bool FROM yolva_data_short"
data = pd.read_sql(query, engine)

# 2. Предобработка данных
data['text'] = data['title'].fillna('') + ' ' + data['description'].fillna('')
data['text'] = data['text'].astype(str)

scaler = StandardScaler()
y_time = scaler.fit_transform(data['original_estimate'].values.reshape(-1, 1))
y_role = data['role_bool'].values

# 3. Токенизация
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def encode_texts(texts, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])

    return np.array(input_ids, dtype=np.int32), np.array(attention_masks, dtype=np.int32)

input_ids, attention_masks = encode_texts(data['text'].tolist(), tokenizer, MAX_LEN)

# 4. Разделение данных
X_ids_train, X_ids_test, X_mask_train, X_mask_test, y_time_train, y_time_test, y_role_train, y_role_test = train_test_split(
    input_ids, attention_masks, y_time, y_time, test_size=0.2, random_state=42
)

# 5. Модель
def create_bert_model():
    bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')

    def bert_layer(inputs):
        input_ids, attention_mask = inputs
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    bert_output = Lambda(
        bert_layer,
        output_shape=(768,),
        name='BERT_Embedding'
    )([input_ids, attention_mask])

    shared = Dense(64, activation='relu')(bert_output)
    shared = Dropout(0.4)(shared)

    time_branch = Dense(32, activation='relu')(shared)
    time_output = Dense(1, name='time_output')(time_branch)

    role_branch = Dense(16, activation='relu')(shared)
    role_output = Dense(1, activation='sigmoid', name='role_output')(role_branch)

    model = Model(inputs=[input_ids, attention_mask], outputs=[time_output, role_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
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


model = create_bert_model()

# 6. Обучение
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    {
        'input_ids': X_ids_train,
        'attention_mask': X_mask_train
    },
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

# 7. Оценка модели
print("\nОценка модели на тестовых данных:")
test_loss = model.evaluate(
    {
        'input_ids': X_ids_test,
        'attention_mask': X_mask_test
    },
    {
        'time_output': y_time_test,
        'role_output': y_role_test
    },
    verbose=0
)

loss_dict = dict(zip(model.metrics_names, test_loss))
print(f"Общий лосс: {loss_dict['loss']:.4f}")
print(loss_dict)
# print(f"Точность выбора роли: {loss_dict['role_output_accuracy'] * 100:.1f}%")

# 8. Предсказание
def predict_task(title, description):
    text = title + ' ' + description

    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    pred_time, pred_role = model.predict(
        {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        },
        verbose=0
    )

    hours = scaler.inverse_transform(pred_time)[0][0]
    role_prob = pred_role[0][0]
    recommendation = 'разработчик' if role_prob > 0.5 else 'аналитик'

    return {
        'title': title,
        'description': description,
        'estimated_hours': round(hours, 1),
        'recommended_role': recommendation,
        'role_confidence': round(float(role_prob if recommendation == 'разработчик' else 1 - role_prob), 3)
    }

# 9. Пример
sample_task = {
    'title': "Обработчик событий на принятие диалога",
    'description': "Обработчик событий по прикреплению диалога и контакта к лиду ..."
}

result = predict_task(**sample_task)
print("\nРезультат предсказания:")
print(f"Задача: {result['title']}")
print(f"Описание: {result['description']}")
print(f"Прогноз времени: {result['estimated_hours']} часов")
print(f"Рекомендуемый исполнитель: {result['recommended_role']} (уверенность: {result['role_confidence'] * 100:.1f}%)")
