# 🔎 Дополнительные углы: редкие события, fairness, deployment, идентичный техностек

> Расширенный поиск по темам, не покрытым в предыдущих файлах.
> Дата: 31.07.2026. Все ссылки ведут на arXiv.

---

## 🎯 Идентичный техностек: XGBoost + LightGBM + CatBoost + Optuna + SHAP

### SHAP-Guided Feature Selection + Optuna + XGBoost/LightGBM/CatBoost
- 📅 **2025-10-22** | 🔗 `arxiv.org/abs/2510.19896`
- **Авторы:** de Oliveira F.F., Rocha M.B., Krohling R.A.
- **Задача:** Диагностика заболеваний мочевыводящих путей
- **Техностек:** XGBoost + LightGBM + CatBoost + **Optuna** (HPO) + **SHAP** (feature selection) + SMOTE (балансировка)
- **Методология:** SHAP используется не только для интерпретации, но и для **отбора признаков** — SHAP-guided feature selection

> 🟢 **Практически твой техностек!** XGBoost, LightGBM, CatBoost, Optuna, SHAP — всё совпадает. Отличие: SHAP используется для feature selection (у тебя — поэтапное экспертное наращивание). Можно сравнить подходы.

### XGBoost + SHAP: Early Mortality Prediction in Elderly MODS (Multicenter)
- 📅 **2020-01-28** | 🔗 `arxiv.org/abs/2001.10977`
- **Авторы:** Liu X., Hu P., Mao Z. et al. (Chinese PLA General Hospital)
- **Данные:** MIMIC-III + eICU-CRD + PLAGH-S (3 базы! Multicenter)
- **Метод:** XGBoost + SHAP
- **Метрики:** AUC, calibration, decision curve analysis

> 🟢 **Multicenter валидация на 3 базах** — золотой стандарт. Твой проект пока на одной когорте; эта статья — образец, как масштабировать на multicenter.

---

## 📉 Редкие события (rare events) — ключевая проблема проекта

### Behavior of Prediction Performance Metrics with Rare Events
- 📅 **2025-04-22** | 🔗 `arxiv.org/abs/2504.16185`
- **Авторы:** Minus E., Coley R.Y., Shortreed S.M., Williamson B.D. (Kaiser Permanente)
- **Вопрос:** AUC может быть **misleading** для редких событий. Что важнее: число событий или event rate?
- **Метрики:** AUC, PPV (positive predictive value), accuracy, sensitivity, specificity
- **Вывод:** ...

> 🔴 **Прямо про твою задачу:** КШ — редкое событие (судя по `scale_pos_weight ≈ 12.7` в XGBoost — ~7-8% prevalence). Эта статья исследует, насколько можно доверять AUC при такой распространённости. Critical read.

### Variational Disentanglement for Rare Event Modeling
- 📅 **2020-09-17** | 🔗 `arxiv.org/abs/2009.08541`
- **Авторы:** Xiu Z., Tao C., Gao M. et al. (Duke)
- **Идея:** Variational disentanglement — полупараметрическое разделение representation learning и classification для rare events
- **Применение:** Healthcare risk prediction с низкой prevalence

> 🟡 Альтернатива `class_weight='balanced'` — disentangled representation вместо коррекции весов. Может дать лучшую калибровку (ср. Sirikul 2026).

---

## ⚖️ Fairness в ICU prediction models

### Monitoring Fairness in ML Models Predicting Patient Mortality in the ICU
- 📅 **2024-10-31** | 🔗 `arxiv.org/abs/2411.00190`
- **Авторы:** van Schaik T.A., Liu X., Atallah L., Badawi O.
- **Fairness dimensions:** Race, sex, medical diagnoses
- **Ключевой инсайт:** Documentation bias — разные группы пациентов имеют разную полноту клинических измерений

> 🟡 **Важно для тебя:** если в твоих данных есть демографические переменные (пол, возраст), модель может быть неодинаково точна для разных групп. Эта статья даёт методологию fairness audit.

### Improving Fairness of LLM-Based ICU Mortality Prediction via Case-Based Prompting
- 📅 **2025-12-17** | 🔗 `arxiv.org/abs/2512.19735`
- **Авторы:** Zhang G., Long Y., Zhou Y., Zhang Y., Hong S.
- **Проблема:** LLM-based prediction exhibits bias по sex, age, race
- **Решение:** Case-based prompting для дебаисинга без деградации accuracy

> 🟢 Тренд 2025-2026: fairness становится обязательным компонентом clinical ML evaluation (наряду с discrimination, calibration).

---

## 🏥 Clinical Deployment / MLOps

### Responsible, Secure and Sustainable Healthcare AI — Strategic Framework
- 📅 **2025-10-09** | 🔗 `arxiv.org/abs/2510.15943`
- **Автор:** Joseph J.
- **5 pillars:** Leadership & Strategy, **MLOps** & Technical Infrastructure, Governance & Ethics, Education, Change Management
- **Пример:** Inpatient LOS prediction (R²=0.41–0.58)

> 🟡 Если думаешь о внедрении модели в клинику — эта статья даёт готовый organisational framework.

---

## 🫀 Cardiac Surgery Complications

### Generative Multi-Task Representation Learning for Postoperative Complications in Cardiac Surgery
- 📅 **2024-12-02** | 🔗 `arxiv.org/abs/2412.01950`
- **Авторы:** Shen J., Xue B., Kannampallil T., Lu C., Abraham J. (Washington University)
- **Метод:** surgVAE (surgical Variational Autoencoder) — cross-task + cross-cohort representation learning
- **Задача:** 6 postoperative complications у cardiac surgery пациентов

> 🟡 Смежная популяция (кардиохирургия). Метод surgVAE может быть полезен, если у тебя multiple outcomes (не только КШ, но и смерть, ОПП, кровотечение).

---

## 🩸 Sepsis Mortality Prediction (методологический аналог)

### Data-Driven ML Approaches for Predicting In-Hospital Sepsis Mortality
- 📅 **2024-08-03** | 🔗 `arxiv.org/abs/2408.01612`
- **Авторы:** Shumilov A., Zhu Y., Ashrafi N. et al. (та же группа Pishgar, что и ICU HF XGBoost)
- **Данные:** MIMIC-III
- **Метод:** literature review + clinical input → feature selection → ML
- **Проблема:** Ограниченная интерпретируемость предыдущих работ → фокус на explainability

> 🟢 Та же исследовательская группа, та же методология, другое заболевание. Полезно для сравнения methodological choices.

---

## 📊 External Validation (Value of Information)

### Value-of-Information Analysis for External Validation of Risk Prediction Models
- 📅 **2026-07-02** | 🔗 `arxiv.org/abs/2607.02321`
- **Авторы:** Wynants L., Wang K.Z., Grimm S. et al. (включая Ewout Steyerberg — автор книги Clinical Prediction Models!)
- **Идея:** EVPI (expected value of perfect information) для оценки, стоит ли проводить дополнительное external validation study
- **Учёт:** Between-center heterogeneity в multicenter studies

> 🟢 **Ewout Steyerberg** — ключевое имя в clinical prediction models. Эта работа расширяет его методологию на multicenter external validation. Если планируешь external validation — must-read.

---

## 🤖 Agentic AI in Medicine (обзор 2026)

### Agentic AI in Medicine: Architectures, Applications, Evaluation, Challenges
- 📅 **2026-07-28** | 🔗 `arxiv.org/abs/2607.25489`
- **Авторы:** Tong Z., Liu Y., Fan W. et al.
- **Тип:** Scoping review (1 649 records screened)
- **Охват:** Planning, tool use, memory, iterative correction, coordination among specialized agents

> 🟢 Обзорная статья (июль 2026) о том, куда движется медицинский AI: от изолированных prediction models → к agentic системам. Полезно для раздела «Future Directions».

---

## 📊 Сводная таблица новых находок

| # | Статья | Дата | Угол | Значимость |
|---|--------|------|------|:----------:|
| 1 | **Rare Events Metrics Behavior** | 2025-04 | AUC для редких событий | 🔴 Critical |
| 2 | **SHAP + Optuna + XGBoost/LGBM/CatBoost** | 2025-10 | Идентичный техностек | 🟢 Высокая |
| 3 | **Fairness Monitoring ICU Mortality** | 2024-10 | Documentation bias | 🟡 Средняя |
| 4 | **LLM Fairness ICU** | 2025-12 | Fairness + дебаисинг | 🟡 Средняя |
| 5 | **surgVAE Cardiac Complications** | 2024-12 | Кардиохирургия | 🟡 Средняя |
| 6 | **XGBoost+SHAP MODS Multicenter** | 2020-01 | 3 базы, multicenter | 🟢 Высокая |
| 7 | **Sepsis Mortality MIMIC-III** | 2024-08 | Аналог методологии | 🟢 Высокая |
| 8 | **Value of Information External Validation** | 2026-07 | Steyerberg, EVPI | 🟢 Высокая |
| 9 | **Healthcare AI MLOps Framework** | 2025-10 | Deployment | 🟡 Средняя |
| 10 | **Agentic AI in Medicine Review** | 2026-07 | Future directions | 🟢 Высокая |
| 11 | **Rare Event Variational Disentanglement** | 2020-09 | Альтернатива class_weight | 🟡 Средняя |

---

## 🎯 Немедленно применимые инсайты

### 1. AUC может быть misleading при малом числе событий (Minus 2025)
Твой prevalence ~7-8%. Проверь:
- Дополнительные метрики: precision-recall AUC (уже есть), PPV, NPV
- Не полагайся только на AUC при сравнении моделей

### 2. SHAP можно использовать для feature selection (de Oliveira 2025)
Вместо поэтапного наращивания (твой подход) можно:
- Обучить модель на всех признаках
- Использовать mean |SHAP| для отбора топ-N
- Сравнить со стадиями — совпадают ли топ-признаки по SHAP с твоими этапами?

### 3. Fairness audit (van Schaik 2024)
Проверить:
- Разное ли качество модели для мужчин/женщин?
- Разное ли для разных классов Killip?
- Есть ли documentation bias (разная полнота данных по группам)?

### 4. Multicenter валидация (Liu 2020)
Образец: MIMIC-III + eICU-CRD + собственная база. Если у тебя одна когорта — external validation на публичной базе (MIMIC, eICU) может усилить статью.
