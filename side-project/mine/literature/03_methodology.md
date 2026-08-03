# ⚙️ Методология: метрики, дисбаланс, импутация

---

## 🔴 1. Коррекция дисбаланса классов: вред или польза?

**Библиография:**
> Sirikul W., Isaradech N., Kiratipaisarl W. et al. (2026). Class Imbalance Corrections Failed to Enhance Discrimination, Model Calibration, and Prediction Stability: An Empirical Simulation Study Based on Clinical Dataset. *arXiv:2606.08966*.

| Параметр | Значение |
|----------|----------|
| Данные | GUSTO-I trial: 40 830 пациентов, 2 851 событий (6.98%) |
| Модель | Penalised logistic regression |
| Методы коррекции | Algorithm-level (class_weight), Data-level (oversampling, combined over+under) |
| Размеры выборок | 500 → 40 830 |
| Валидация | 200 bootstrap resamples |
| Метрики | AUC, калибровка, calibration stability, MAPE, CII |
| Ссылка | https://arxiv.org/abs/2606.08966 |

### Основные результаты
1. **Дискриминация (AUC):** коррекция дисбаланса **не дала значимого улучшения** ни при каком размере выборки
2. **Калибровка:** все методы коррекции привели к **miscalibration** — модель систематически завышала риск
3. **Стабильность:** повышение prediction instability (MAPE, CII)

### Вывод авторов (прямая цитата)
> *«Class imbalance should not be treated as a pathology that automatically requires correction. In clinical prediction modelling, routine imbalance correction by default is generally not advisable.»*

### Что это значит для нашего проекта

В `baseline.ipynb` используются:
```python
# Random Forest
class_weight='balanced_subsample'

# XGBoost
scale_pos_weight=12.675771709705893

# LightGBM
class_weight='balanced'

# CatBoost
auto_class_weights='Balanced'
```

**Рекомендуемое действие:**
- Провести **ablation study**: обучить все 4 модели без коррекции дисбаланса
- Сравнить: AUC, Brier score, calibration curves
- Ожидаемый результат (по Sirikul 2026): AUC не изменится, но калибровка улучшится

---

## 🟡 2. MICE vs детерминированная импутация для prediction models

**Библиография:**
> Mi J., Tendulkar R.D., Sittenfeld S.M.C. et al. (2024). Combining Missing Data Imputation and Internal Validation in Clinical Risk Prediction Models. *arXiv:2411.14542*.

| Параметр | Значение |
|----------|----------|
| Контекст | Clinical risk prediction models (не estimation/descriptive studies) |
| Аргумент | Детерминированная импутация лучше подходит для prediction |
| Причина | Target не включается в модель импутации → легко применить к новым пациентам |
| Метод | Bootstrapping + детерминированная импутация |
| Ссылка | https://arxiv.org/abs/2411.14542 |

### Ключевые различия: estimation vs prediction

| | Estimation models | Prediction models |
|---|---|---|
| **Цель** | Оценить причинно-следственную связь | Максимизировать точность прогноза |
| **Импутация** | MICE (включает outcome) | Детерминированная (НЕ включает outcome) |
| **Production** | Не предполагается | Должна работать на новых пациентах |

### Что это значит для нашего проекта

В `baseline.ipynb` MICE используется правильно — с `variable_schema`, где target не используется для импутации предикторов. Однако:
- MICE требует итеративной подгонки → медленно в production
- Mi et al. рекомендуют bootstrapping + **SimpleImputer** (mean/median) для prediction

**Рекомендуемое действие:**
- Сравнить качество моделей с MICE vs SimpleImputer
- Если разница в AUC небольшая, детерминированная импутация предпочтительнее для production

---

## 🟢 3. Brier Score: правильная интерпретация

### 3a. Weighted Brier Score for Clinical Utility

**Библиография:**
> Zhu K., Zheng Y., Chan K.C.G. (2024). Weighted Brier Score — an Overall Summary Measure for Risk Prediction Models with Clinical Utility Consideration. *arXiv:2408.01626*.

| Параметр | Значение |
|----------|----------|
| Проблема | Обычный Brier score не учитывает clinical utility |
| Решение | **Weighted Brier score**, согласованный с decision-theoretic framework |
| Декомпозиция | Discrimination + calibration компоненты |
| Связь | Weighted Brier ↔ H-measure (альтернатива AUC) |
| Ссылка | https://arxiv.org/abs/2408.01626 |

### 3b. Misconceptions about the Brier Score

**Библиография:**
> Hoessly L. (2025). On Misconceptions about the Brier Score in Binary Prediction Models. *arXiv:2504.04906*.

| Параметр | Значение |
|----------|----------|
| Проблема | Brier score часто неправильно интерпретируется в клинических исследованиях |
| Причина | Brier score не согласуется с традиционными концептами medical statistics |
| Ссылка | https://arxiv.org/abs/2504.04906 |

### Что это значит для нашего проекта

В `baseline.ipynb` Brier score уже используется — это правильно. Рекомендации:
1. При интерпретации Brier score учитывать prevalence (долю положительного класса) — Hoessly (2025)
2. Рассмотреть **weighted Brier score** как дополнительную метрику — Zhu (2024)
3. Декомпозировать Brier score на discrimination (разброс предсказаний) и calibration (смещение) компоненты

---

## 📊 Резюме: recommended actions

| # | Действие | Обоснование | Приоритет |
|---|----------|-------------|:---------:|
| 1 | Ablation study без `class_weight='balanced'` | Sirikul 2026: коррекция вредит калибровке | 🔴 Высокий |
| 2 | Сравнить MICE vs SimpleImputer | Mi 2024: детерминированная импутация для production | 🟡 Средний |
| 3 | Добавить CardShock score как baseline | Hu 2023: стандарт в литературе (AUROC 0.519) | 🟡 Средний |
| 4 | Weighted Brier score | Zhu 2024: лучше отражает clinical utility | 🟢 Низкий |
| 5 | ALE plots (альтернатива SHAP) | Fan 2025: менее чувствительны к корреляции | 🟢 Низкий |

---

## 📖 Дополнительные источники (не на arXiv)

Эти работы упоминаются в найденных статьях и релевантны проекту:

- **Harjola V.P. et al. (2015).** CardShock Study — валидация CardShock risk score. *European Journal of Heart Failure*.
- **Thiele H. et al. (2012).** IABP-SHOCK II Trial — внутриаортальная баллонная контрпульсация при КШ. *NEJM*.
- **Steyerberg E.W. et al. (2010).** Clinical Prediction Models — книга-стандарт по разработке и валидации prediction models.
- **Van Buuren S. (2018).** Flexible Imputation of Missing Data — книга по MICE, используется в `miceforest`.
- **Lundberg S.M., Lee S.I. (2017).** SHAP: A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
