# 🏥 ML в ICU: методологические аналоги

---

## 1. XGBoost: Mortality Prediction for ICU Heart Failure Patients (MIMIC-III)

**Библиография:**
> Ashrafi N., Abdollahi A., Zhang J., Pishgar M. (2024). Optimizing Mortality Prediction for ICU Heart Failure Patients: Leveraging XGBoost and Advanced Machine Learning with the MIMIC-III Database. *arXiv:2409.01685*.

| Параметр | Значение |
|----------|----------|
| Задача | Предсказание смертности пациентов с СН в ICU |
| Данные | MIMIC-III, n=1 177 |
| Модель | **XGBoost** (лучшая из LogReg, SVM, RF, LGBM, XGBoost) |
| AUROC (test) | **0.9228** (95% CI 0.8748–0.9613) |
| Feature selection | VIF + клиническая экспертиза + ablation studies → 46 признаков |
| Imbalance | Oversampling |
| HP tuning | Grid-Search |
| Интерпретация | **SHAP** + feature importance |
| Ключевые признаки | Leucocyte count, RDW |
| Ссылка | https://arxiv.org/abs/2409.01685 |

### Сходство с нашим проектом
- Та же линейка моделей (LogReg, RF, XGBoost, LGBM)
- **SHAP-интерпретация** топ-признаков
- Продвинутый feature selection (у нас — экспертное поэтапное наращивание)
- Oversampling для дисбаланса классов (спорная практика — см. `03_methodology.md`)

### Отличия
- MIMIC-III (публичная база) vs наши данные
- Grid-Search vs **Optuna** (у нас — более эффективный подбор)
- 46 признаков vs поэтапное наращивание 3→6→9→10

### Что можно позаимствовать
- VIF (Variance Inflation Factor) для проверки мультиколлинеарности предикторов
- Ablation studies: поочерёдное удаление групп признаков и замер падения AUC

---

## 2. XMI-ICU: Explainable ML for Pseudo-Dynamic Mortality Prediction (Heart Attack)

**Библиография:**
> Mesinovic M., Watkinson P., Zhu T. (2023). XMI-ICU: Explainable Machine Learning Model for Pseudo-Dynamic Prediction of Mortality in the ICU for Heart Attack Patients. *arXiv:2305.06109*.

| Параметр | Значение |
|----------|----------|
| Задача | Предсказание смертности при ОИМ в ICU |
| Данные | eICU + MIMIC-IV (внешняя валидация) |
| Модель | XGBoost |
| AUROC | **0.91** (balanced accuracy 82.3) за 6 часов до события |
| Особенность | **Time-resolved SHAP** — интерпретация с временной динамикой |
| Ссылка | https://arxiv.org/abs/2305.06109 |

### Ключевая инновация
**Pseudo-dynamic framework**: временные ряды физиологических измерений преобразуются в stacked static prediction problems. Это позволяет:
- Делать предсказания с окном до 24 часов
- Получать **time-resolved SHAP values** — как вклад каждого признака меняется со временем

### Идея для проекта
Если в данных есть временные срезы (а в реанимации они обычно есть — показатели снимаются каждые 1–4 часа), можно адаптировать pseudo-dynamic подход для предсказания КШ за N часов до развития.

---

## 3. CatBoost: Elderly ICU Patients with Diabetes and Heart Failure

**Библиография:**
> Fan J., Chen S., Sun L. et al. (2025). Predicting Short-Term Mortality in Elderly ICU Patients with Diabetes and Heart Failure: A Distributional Inference Framework. *arXiv:2506.15058*.

| Параметр | Значение |
|----------|----------|
| Задача | Краткосрочная смертность у пожилых ICU пациентов с СД + СН |
| Данные | MIMIC-IV, n=1 478, 19 признаков |
| Модель | **CatBoost** (лучшая из 6 моделей) |
| AUROC | **0.863** |
| Особенность | **DREAM-алгоритм** — posterior distributions вместо точечных оценок |
| Интерпретация | Ablation + **ALE (Accumulated Local Effects)** plots |
| Ключевые признаки | APS III, oxygen flow, GCS eye, Braden Mobility |
| Ссылка | https://arxiv.org/abs/2506.15058 |

### Почему это важно для нас
- **CatBoost — лучшая модель** на клинических данных с категориальными признаками (подтверждает наш выбор)
- ALE plots — альтернатива SHAP dependency plots, менее чувствительная к корреляции признаков
- DREAM для uncertainty quantification — можно применить у себя

---

## Сравнительная таблица методологий

| Элемент | Ashrafi 2024 | Mesinovic 2023 | Fan 2025 | **Наш проект** |
|---------|:-----------:|:-------------:|:-------:|:------------:|
| Модель | XGBoost | XGBoost | CatBoost | 5 моделей |
| Данные | MIMIC-III | eICU+MIMIC-IV | MIMIC-IV | Собственные |
| HP tuning | Grid-Search | — | — | **Optuna** |
| SHAP | ✓ | ✓ (time-resolved) | — | ✓ (CatBoost fix) |
| Импутация | — | — | — | **MICE** |
| Этапы признаков | — | — | — | **4 этапа** |
| Калибровка | — | — | — | **Brier + curves** |
| DCA | — | — | — | ✓ (функция есть) |
