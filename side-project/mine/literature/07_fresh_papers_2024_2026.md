# 🆕 Свежие статьи (2024–2026)

> Отобраны по релевантности к проекту (КШ + ML + ICU + методология).
> Все статьи с arXiv, дата поиска: 31.07.2026.

---

## 🫀 Кардиогенный шок + ML

### Guardian-regularized Safe Offline RL for Weaning of MCS in Cardiogenic Shock
- 📅 **2025-11-08** | 🔗 `arxiv.org/abs/2511.06111`
- **Авторы:** Tumay A., Sun S., Fereidooni S., Dumas A., Jortberg E., Yu R. (UC San Diego)
- **Задача:** Автоматизированное отключение MCS (механическая поддержка кровообращения) у пациентов с **кардиогенным шоком**
- **Метод:** Offline RL с «Guardian»-регуляризацией для безопасного принятия решений
- **Контекст:** MCS (Impella и аналоги) — микроаксиальные насосы для разгрузки левого желудочка. Текущие стратегии отлучения варьируются между клиниками, нет data-driven подхода.

> 🟢 **Прямая связь с проектом:** та же популяция (КШ), смежная задача (управление лечением, а не предсказание). Можно обсудить в related work как complementary подход.

---

## 🎯 Conformal Prediction + Gradient Boosting (методологически важно)

### Conformal Risk Prediction for NAFLD Using Gradient Boosting with Distribution-Free Coverages
- 📅 **2026-05-31** | 🔗 `arxiv.org/abs/2606.09860`
- **Автор:** Zhang X.
- **Метод:** Gradient-boosted decision trees + **conformal prediction** для calibrated risk estimates
- **Feature selection:** Mutual-information-based stability selection через bootstrap
- **Результат:** Distribution-free coverage guarantees на индивидуальных risk estimates

> 🟢 **Идеальный methodological template для твоего проекта:** gradient boosting (как у тебя) + conformal prediction (рекомендация из `06_uncertainty_survival.md`) + stability selection для компактного набора признаков. Практически рецепт: возьми CatBoost, оберни в conformal predictor, получи calibrated prediction intervals.

---

## 🏥 ICU + Импутация

### Closing Gaps: An Imputation Analysis of ICU Vital Signs
- 📅 **2025-10-28** | 🔗 `arxiv.org/abs/2510.24217`
- **Авторы:** Turubayev A., Shopova A., Lange F. et al.
- **Задача:** Сравнение методов импутации для ICU vital signs (heart rate, etc.) с большими пропусками
- **Ключевой вывод:** Нужна более comprehensive comparison импутационных методов на ICU данных — существующие работы фрагментарны

> 🟡 **Рекомендация:** эта статья — хороший ориентир для сравнения твоего MICE-подхода с альтернативами. Если у тебя ещё нет comparison с SimpleImputer, median, KNN — эта работа даёт методологический шаблон.

---

## 🧱 Tabular Foundation Models (тренд 2026)

### Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation
- 📅 **2026-06-10** | 🔗 `arxiv.org/abs/2606.12006`
- **Авторы:** Pham M.K., Cotugno L., Sirbu A. et al.
- **Идея:** Tabular foundation models (предобученные на табличных данных) + survival analysis для клинического time-to-event prediction
- **Проблема:** Стандартные tabular FM работают на дискретных классах, но не поддерживают censored time-to-event данные
- **Решение:** Survival-aware adaptation — адаптация foundation model к цензурированным данным

### Retrieval-aligned Tabular Foundation Models Enable Robust Clinical Risk Prediction in EHR
- 📅 **2026-04-02** | 🔗 `arxiv.org/abs/2604.01841`
- **Авторы:** Pham M.K., Nguyen Ho T.L., Dao T.T.P. et al.
- **Бенчмарк:** Multi-cohort EHR — сравнение classical, deep tabular и TICL (tabular in-context learning) моделей
- **Условия:** Разный масштаб данных, размерность признаков, редкость исходов, cross-cohort generalization
- **Ключевой вывод:** PFN-based TICL модели sample-efficient при малых данных, но **деградируют при class imbalance и distribution shift**

> 🟡 **Почему это важно:** ты работаешь с imbalanced clinical данными — этот бенчмарк валидирует, что классические методы (твой CatBoost/XGBoost) могут быть надёжнее модных foundation models в условиях дисбаланса.

### Tabular LLMs for Interpretable Few-Shot Alzheimer's Disease Prediction
- 📅 **2026-03-17** | 🔗 `arxiv.org/abs/2603.17191`
- **Авторы:** Kearney S., Yang S., Wen Z. et al.
- **Идея:** LLM-based подход к табличным клиническим данным: TAP-GPT (на базе TableGPT2)
- **Результат:** Few-shot clinical prediction с интерпретируемыми output

> 🟢 **Future direction:** LLM для табличных клинических данных — растущий тренд 2026. Пока рано для production, но интересно для discussion.

---

## 📊 Трансферное обучение + пропуски

### Distributionally Robust Transfer Learning with Structurally Missing Covariates (DRUM) — Cardiac Arrest
- 📅 **2026-05-22** | 🔗 `arxiv.org/abs/2605.24212`
- **Авторы:** Li S., Hong C., Tian Z. et al. (Harvard, Stanford)
- **Задача:** Перенос модели предсказания cardiac arrest между странами с **разным набором доступных признаков**
- **Проблема:** Модели, обученные в high-resource settings, используют признаки, недоступные в других регистрах
- **Метод:** DRUM — distributionally robust transfer learning с учётом структурно отсутствующих ковариат

> 🟡 **Актуально:** если твоя модель будет внедряться в разных клиниках с разным набором доступных предикторов (где-то есть SII, где-то нет), DRUM — это метод для решения такой проблемы.

---

## 🩺 Клинические risk scores (аналогичные твоему)

### Improving Risk Stratification in Hypertrophic Cardiomyopathy (HCM)
- 📅 **2026-03-27** | 🔗 `arxiv.org/abs/2603.26254`
- **Авторы:** Taconné M., Corino V.D.A., Del Franco A. et al.
- **Задача:** ML risk score для ГКМП на основе ЭхоКГ + клинических + медикаментозных данных
- **Сравнение:** Против ESC score (аналог твоего сравнения с CardShock/TIMI)
- **Метод:** Объяснимый ML (как у тебя SHAP)

### Comprehensive Evaluation of ML for T2D Risk: External Validation + Fairness
- 📅 **2026-06-27** | 🔗 `arxiv.org/abs/2607.16253`
- **Авторы:** Pall R.S., Yadav S., Bhalerao S. et al.
- **Методология:** Multi-dimensional framework: **discrimination + calibration + interpretability + fairness**
- **Модель:** XGBoost на NHANES (n=15 685), внешняя валидация
- **Проблема:** Хорошие internal результаты разрушаются без external testing

> 🟢 **Методологический шаблон:** эта работа — пример rigorous evaluation framework. Оценка не только AUC, но и калибровка + интерпретация + fairness + внешняя валидация. Твой проект уже близок к этому стандарту (есть AUC, PR-AUC, Brier, калибровка, SHAP).

---

## 📈 Uncertainty + Биомаркеры

### Uncertainty-Calibrated Prediction of Randomly-Timed Biomarker Trajectories with Conformal Bands
- 📅 **2025-11-17** | 🔗 `arxiv.org/abs/2511.13911`
- **Авторы:** Tassopoulou V., Stamouli C., Shou H., Pappas G.J., Davatzikos C. (UPenn)
- **Задача:** Conformal prediction bands для траекторий биомаркеров при нерегулярных визитах
- **Инновация:** Nonconformity score для randomly-timed trajectories
- **Гарантии:** Prediction bands гарантированно покрывают истинную траекторию

> 🟢 **Перспектива:** если добавишь в проект временную динамику биомаркеров (ЧСС, SpO2 во времени), этот метод даёт способ строить prediction bands вместо точечных прогнозов — напрямую в клиническую практику.

---

## 📊 Хронологическая сводка (2024–2026)

```
2024 ───┬─ Feb: Self-Calibrating Conformal Prediction (van der Laan)
        ├─ May: Beyond Random Missingness (Qian)
        ├─ Jul: Tabular Data Augmentation survey (Cui)
        ├─ Aug: Weighted Brier Score (Zhu)
        ├─ Sep: CTLPE Irregular TS (Kim) | XGBoost ICU HF MIMIC-III (Ashrafi)
        ├─ Oct: Deep Survival Analysis monograph (Chen)
        ├─ Nov: MICE vs Deterministic Imputation (Mi)
        └─ Dec: MICE-RF vs DL for Healthcare TS (Le) | ECG Liver Diagnosis (Lopez Alcaraz)

2025 ───┬─ Apr: Misconceptions about Brier Score (Hoessly)
        ├─ May: ML + SHAP Osteoporosis (Elias) 
        ├─ Jun: Elderly ICU HF+DM CatBoost (Fan)
        ├─ Oct: Closing Gaps ICU Imputation (Turubayev)
        ├─ Nov: Conformal Biomarker Trajectories (Tassopoulou) | Guardian RL Cardiogenic Shock (Tumay)
        └─ Dec: XAI vs Linear Regression Lung Cancer (Hashtarkhani)

2026 ───┬─ Jan: MDS-ICU Multimodal Deterioration (Lopez Alcaraz)
        ├─ Feb: STAR-Set Async Clinical TS (Lee)
        ├─ Mar: Tabular LLMs Alzheimer (Kearney) | HCM Risk Score (Taconné)
        ├─ Apr: Retrieval-aligned Tabular FM EHR (Pham)
        ├─ May: DRUM Transfer Learning Cardiac Arrest (Li) | Conformal NAFLD Gradient Boosting (Zhang)
        ├─ Jun: Tabular FM Survival (Pham) | T2D External Validation Fairness (Pall) | Class Imbalance Harm (Sirikul)
        └─ Jul: EHR-RAGp Foundation Model (Shurrab)
```

---

## 🎯 Топ-5 свежих статей для немедленного применения

| # | Статья | Дата | Что даёт проекту |
|---|--------|------|------------------|
| 1 | **Conformal NAFLD Gradient Boosting** | 2026-05 | Рецепт: GB + conformal prediction на клинических данных |
| 2 | **DRUM Transfer Learning Cardiac Arrest** | 2026-05 | Как предсказывать при неполном наборе признаков |
| 3 | **Closing Gaps ICU Imputation** | 2025-10 | Методология сравнения импутаций для ICU |
| 4 | **T2D External Validation + Fairness** | 2026-06 | Multi-dimensional evaluation framework |
| 5 | **Guardian RL Cardiogenic Shock** | 2025-11 | Та же популяция — можно цитировать как смежную работу |
