import re
import pandas as pd
from bs4 import BeautifulSoup

def clean_html(text):
    if pd.isna(text):
        return text

    # Удаляем HTML-теги с помощью BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)

    # Заменяем множественные пробелы и переносы строк на одинарные пробелы
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # Удаляем оставшиеся спецсимволы (кроме обычных пунктуационных знаков)
    clean_text = re.sub(r'[^\w\s.,;:!?()-]', ' ', clean_text)

    # Еще раз убираем лишние пробелы
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


# Список файлов для обработки
files_to_process = [
    'schneider_in.csv',
    'deluxephoto.csv',
    'innerCRM.csv',
    'scolcovo.csv',
    'shneider_vnedreniye.csv',
    't1_cloud.csv',
    'tikkurila.csv',
    'tp_schn.csv',
    'tp_schneider.csv',
    'tp_tikkurila.csv',
    'usergate.csv',
    'vtbl.csv'
]

for input_file in files_to_process:
    try:
        # Загрузка CSV файла
        df = pd.read_csv(input_file)

        # Проверяем наличие столбца 'Description' (если его нет, пропускаем)
        if 'Description' in df.columns:
            # Очистка столбца description
            df['Description'] = df['Description'].apply(clean_html)

            # Формируем имя выходного файла (добавляем префикс cleaned_)
            output_file = f'cleaned_{input_file}'

            # Сохранение результата
            df.to_csv(output_file, index=False)
            print(f"Файл {input_file} успешно обработан. Результат сохранён в {output_file}")
        else:
            print(f"Файл {input_file} не содержит столбца 'Description'. Пропускаем.")
    except Exception as e:
        print(f"Ошибка при обработке файла {input_file}: {e}")

print("Обработка всех файлов завершена.")