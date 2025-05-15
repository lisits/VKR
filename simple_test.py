from simple_predictor import PredictorSimple

if __name__ == "__main__":
    sample_task = {
        "title": "Импорт исторических данных - INSTALLED_PRODUCT (Оборудование)",
        "description": "Разработать скрипт для импорта записей из xlsx (csv) таблицы в смарт-процесс INSTALLED_PRODUCT (см. вложения)"
    }

    predictor = PredictorSimple()
    result = predictor.predict_task(**sample_task)

    print("\nРезультат предсказания:")
    print(f"Задача: {result['title']}")
    print(f"Описание: {result['description']}")
    print(f"Прогноз времени: {result['estimated_hours']} часов")
    print(f"Рекомендуемый исполнитель: {result['recommended_role']} (уверенность: {result['role_confidence'] * 100:.1f}%)")
