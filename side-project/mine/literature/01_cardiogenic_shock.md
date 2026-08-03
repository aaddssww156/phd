# 🫀 Прямые аналоги: предсказание кардиогенного шока (КШ)

---

## 1. CShock: A Dynamic Risk Score for Early Prediction of Cardiogenic Shock Using Machine Learning

**Библиография:**
> Hu Y., Lui A., Goldstein M., Sudarshan M. et al. (2023). A dynamic risk score for early prediction of cardiogenic shock using machine learning. *arXiv:2303.12888*.

| Параметр | Значение |
|----------|----------|
| Задача | Предсказание КШ у пациентов cardiac ICU с ОСН/ОИМ |
| Данные | Аннотированные врачом cardiac ICU datasets |
| Метод | Deep Learning |
| AUROC (внутр.) | **0.820** |
| AUROC (внешн.) | **0.800** |
| Бенчмарк | CardShock score: AUROC 0.519 |
| Ссылка | https://arxiv.org/abs/2303.12888 |

### Ключевые моменты
- **Та же клиническая задача**, что и в нашем проекте
- Использует глубокое обучение (DL), а не градиентный бустинг
- **CardShock score** — стандартный риск-инструмент, который CShock превзошёл в 1.5 раза
- Внешняя валидация на независимой когорте — сильный аргумент в пользу generalizability

### Сравнение с нашим проектом
| Аспект | CShock (Hu 2023) | Наш проект |
|--------|-------------------|------------|
| Модель | Deep Learning | LogReg, RF, XGBoost, LGBM, CatBoost |
| Предикторы | Все доступные в EHR | Поэтапное наращивание (4 этапа) |
| Интерпретация | Нет | SHAP (summary, bar, dependence, waterfall) |
| Импутация | Не указана | MICE (LightGBM) |
| Калибровка | Нет | Calibration curves, Brier score |
| Бенчмарк | CardShock (AUROC 0.519) | Пока нет |

### Идеи для проекта
1. **Добавить CardShock score как baseline** — это стандарт в литературе
2. Обсудить, почему бустинги могут быть лучше DL на малых выборках
3. Подчеркнуть SHAP-интерпретацию как преимущество перед CShock

---

## 2. CardShock Risk Score (базовый бенчмарк)

**Оригинальная работа:**
> Harjola V.P., Lassus J., Sionis A. et al. (2015). Clinical picture and risk prediction of short-term mortality in cardiogenic shock. *European Journal of Heart Failure*, 17(5), 501–509.

**Предикторы CardShock score:**
1. Возраст > 75 лет
2. Систолическое АД < 80 mmHg
3. ЧСС > 100 уд/мин
4. Способность к самостоятельному дыханию (confusion)
5. ФВ ЛЖ < 40%
6. Предшествующий ИМ или АКШ
7. Лактат > 4 ммоль/л
8. Скорость клубочковой фильтрации (eGFR)

В CShock paper этот скоринговый инструмент показал AUROC **0.519** для раннего предсказания КШ — практически на уровне случайного угадывания, что подчёркивает сложность задачи.

---

## Контекст: почему задача сложная

Цитата из CShock paper:
> *«Early recognition of cardiogenic shock is critical. Prompt implementation of treatment measures can prevent the deleterious spiral of ischemia, low blood pressure, and reduced cardiac output due to cardiogenic shock. However, early identification of cardiogenic shock has been challenging due to human providers' inability to process the enormous amount of data in the cardiac ICU and lack of an effective risk stratification tool.»*

Это прямое обоснование актуальности нашего проекта: существующие шкалы (CardShock, TIMI, GRACE) недостаточно точны для раннего предсказания КШ, и ML-модели на рутинных клинических данных могут заполнить этот пробел.
