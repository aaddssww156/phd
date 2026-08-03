# 🧩 Расширенный охват: AutoML, федеративное обучение, causal inference, early prediction

> Сборник статей по темам, выходящим за рамки основной методологии, но критически важным
> для масштабирования и clinical deployment проекта.

---

## 1. 🤖 AutoML для клинических табличных данных

### AutoML-Med: Automated ML Framework for Medical Tabular Data
- 📅 **2025-08-04** | 🔗 `arxiv.org/abs/2508.02625`
- **Авторы:** Francia R., Leone M., Leonardi G. et al.
- **Проблемы медицинских данных:** missing values, class imbalance, heterogeneous feature types, high dimensionality vs малый размер выборки
- **Решение:** AutoML-Med — автоматический подбор preprocessing + model с минимумом user intervention
- **Архитектура:** Latin Hypercube Sampling для поиска оптимальной комбинации

> 🟢 **Прямая связь:** твой проект делает preprocessing (MICE, scaling) + model selection (5 моделей) + hyperparameter tuning (Optuna) вручную. AutoML-Med автоматизирует этот процесс и может служить baseline для сравнения с твоим ручным pipeline.

### CliMB: AI-enabled Partner for Clinical Predictive Modeling
- 📅 **2024-09-30** | 🔗 `arxiv.org/abs/2410.03736`
- **Авторы:** Saveliev E., Schubert T., Pouplin T., Kosmoliaptsis V., **van der Schaar M.** (Cambridge/Oxford)
- **Проблема:** «Domain expert-AI conundrum» — врачи не могут использовать SOTA ML-инструменты
- **Решение:** CliMB — партнёр для клинициста: AutoML + interpretability + clinical workflow integration

> 🟢 **van der Schaar** — ведущий исследователь ML for healthcare. CliMB — это видение того, как clinical prediction models должны создаваться врачами, а не data scientist'ами.

---

## 2. 🏥 Early Prediction (первые 24 часа в ICU)

### Early Prediction of In-Hospital ICU Mortality: A Review
- 📅 **2025-05-18** | 🔗 `arxiv.org/abs/2505.12344`
- **Авторы:** Huang B., Chen C., Hou X. et al. (19 авторов)
- **Тип:** Systematic review — early prediction в первые 24 часа ICU
- **Фокус:** ML, novel biomarkers, integration подходы
- **Вывод:** Традиционные scoring systems (APACHE, SOFA) ограничены → ML превосходит

> 🟢 **Прямо про твою задачу:** предсказание по данным первых 24 часов (твои Killip, ЧСС, SpO2, глюкоза — это именно первые измерения при поступлении). Review даёт ландшафт методов.

### Multimodal Mortality Prediction: Multicenter External Validation
- 📅 **2025-12-15** | 🔗 `arxiv.org/abs/2512.19716`
- **Авторы:** Mamandipoor B., Hsu C.N., Krause M., Schmidt U.H., Gabriel R.A. (UCSD)
- **Данные:** MIMIC-III + MIMIC-IV + **eICU + HiRID** (4 базы!)
- **Модальности:** Structured (time series, первые 24h) + unstructured text
- **Задача:** In-hospital mortality после первых 24 часов ICU

> 🟢 **Образец multicenter external validation** на 4 базах. Если планируешь external validation — это methodological gold standard.

### Early 30-Day Mortality: Hypertension + AF in ICU
- 📅 **2025-06-18** | 🔗 `arxiv.org/abs/2506.15036`
- **Авторы:** Chen S., Si Y., Fan J. et al. (группа Pishgar)
- **Данные:** MIMIC-IV, 1 301 пациент с гипертонией + AF
- **Признаки:** Chart events, labs, procedures, medications, demographics (первые 24h)
- **Результат:** 17 клинических переменных после feature selection
- **Методы:** MICE импутация, feature selection, ML

> 🟢 **Методологический близнец:** MICE → feature selection → ML на данных первых 24 часов. Та же группа (Pishgar), те же методы.

---

## 3. 🔒 Федеративное обучение (Federated Learning)

### FLICU: Federated Learning for ICU Mortality Prediction
- 📅 **2022-05-30** | 🔗 `arxiv.org/abs/2205.15104`
- **Авторы:** Mondrejevski L., Miliou I., Montanino A. et al. (Stockholm University)
- **Проблема:** Healthcare data чувствительны, хранятся в data silos → нельзя объединить
- **Решение:** FLICU — federated learning workflow для ICU mortality prediction

> 🟡 Если данные из нескольких клиник нельзя объединить (юридически/этически) — federated learning позволяет обучить модель на всех данных без их перемещения.

### Federated Learning: Multicenter Postoperative Complications (OneFlorida+)
- 📅 **2026-03-17** | 🔗 `arxiv.org/abs/2603.16723`
- **Авторы:** Ren Y., Vemuri V.S., Hu Z. et al. (University of Florida)
- **Данные:** 358 644 пациентов, 494 163 хирургических процедур, 5 госпиталей
- **Результат:** FL модели дают robust generalizability при сохранении privacy

> 🟢 **Реальный multicenter FL** на сотнях тысяч пациентов — production-grade подход.

---

## 4. 🔬 Causal Inference vs Prediction

### Causal Inference in Medicine and Health Policy (обзор)
- 📅 **2021-05-10** | 🔗 `arxiv.org/abs/2105.04655`
- **Авторы:** Zhang W., Ramezani R., Naeim A. (UCLA)
- **Ключевой тезис:** «Healthcare practitioners are not content with mere predictions — they are also interested in the cause-effect relation between input features and clinical outcomes»

> 🟡 **Почему это важно:** твой SHAP-анализ показывает association, но не causation. SII повышен → выше риск КШ? Или КШ вызывает повышение SII? Causal inference методы (do-calculus, instrumental variables) позволяют различить.

### Incorporating Causal Effects into DL Predictions on EHR
- 📅 **2020-11-11** | 🔗 `arxiv.org/abs/2011.05466`
- **Авторы:** Li J., Yang H., Jia X., Kumar V., Steinbach M., Simon G. (UMN)
- **Метод:** Квантификация clinically well-defined causal effects → incorporation в DL модели
- **Проблема:** EHR данные имеют сложную causal structure → naive DL делает biased predictions

> 🟢 Если хочешь усилить дискуссию о причинно-следственных связях (почему именно эти предикторы важны) — causal inference методы дают более строгую рамку, чем SHAP.

---

## 5. 📊 Dataset Shift / Rashomon Effect / Generalization

### Intervention Efficiency & Perturbation Validation: Rashomon Effect
- 📅 **2025-11-18** | 🔗 `arxiv.org/abs/2511.14317`
- **Авторы:** Zhang Y., Tran V., Weng P.
- **Проблема:** Rashomon Effect — множество моделей с comparable performance, но разными решениями
- **Решение:** Capacity-aware model selection framework с perturbation validation

> 🔴 **Прямо про твой случай:** у тебя 5 моделей (LogReg, RF, XGBoost, LGBM, CatBoost) могут иметь близкий AUC, но разные SHAP-рейтинги признаков. Rashomon Effect — формальная рамка для выбора между ними.

### Understanding Behavior of Clinical Models under Domain Shifts
- 📅 **2018-09-20** | 🔗 `arxiv.org/abs/1809.07806`
- **Авторы:** Thiagarajan J.J., Rajan D., Sattigeri P. (LLNL)
- **Проблема:** Модели деградируют при переносе между госпиталями (разные протоколы, популяции)

### Domain-invariant Clinical Representation Learning
- 📅 **2023-10-11** | 🔗 `arxiv.org/abs/2310.07799`
- **Авторы:** Zhang Z., Wang Y., Zhu Y. et al.
- **Решение:** Domain-invariant representations — модель, устойчивая к смене датасета

---

## 6. 🎲 Bayesian Uncertainty для клинических prediction

### Inadequacy of Stochastic Neural Networks for Reliable Clinical Decision Support
- 📅 **2024-01-24** | 🔗 `arxiv.org/abs/2401.13657`
- **Авторы:** Lindenmeyer A., Blattmann M., Franke S., Neumuth T., Schneider D. (Universität Leipzig)
- **Проблема:** Common DL approaches overconfident under data shift
- **Вывод:** «Common stochastic neural networks are inadequate» — нужны более надёжные методы uncertainty quantification

> 🔴 **Важно:** эта статья утверждает, что стандартные stochastic NN (MC Dropout, ensembles) недостаточно надёжны для клиники. Conformal prediction (см. `06_uncertainty_survival.md`) предлагает более строгие гарантии.

### Deep Bayesian Gaussian Processes for Uncertainty in EHR
- 📅 **2020-03-23** | 🔗 `arxiv.org/abs/2003.10170`
- **Авторы:** Li Y., Rao S., Hassaine A. et al. (Oxford, UK Biobank)
- **Метод:** Deep Bayesian Gaussian Processes — более выразительная альтернатива Bayesian NN
- **Проблема:** Bayesian NN — lack of expressiveness; Deep Kernel Learning — captures only higher-level uncertainty

> 🟡 Альтернативный метод uncertainty quantification для клинических prediction — между простым MC Dropout и сложным conformal prediction.

---

## 7. 🏗️ Model Compression / Knowledge Distillation

### OrthKD: Extracting Generalized Clinical Knowledge for Lightweight Deployment
- 📅 **2026-07-28** | 🔗 `arxiv.org/abs/2607.25545`
- **Авторы:** Xu Y., Chen C., Cao M.
- **Задача:** Multi-teacher knowledge distillation для diabetic retinopathy screening на edge devices
- **Инсайт:** KD позволяет уменьшить модель в разы без потери точности

> 🟡 Если твой CatBoost (700 итераций, depth=4) нужно запускать на CPU в клинике — KD позволяет сжать до меньшей модели.

---

## 8. 🫀 Специфичные предикторы

### Comparative Study of ML Algorithms in Detecting Cardiovascular Diseases
- 📅 **2024-05-27** | 🔗 `arxiv.org/abs/2405.17059`
- **Авторы:** Dayana K., Nandini S., Varshini R.S.
- **Сравнение:** Multiple ML algorithms для детекции CVD

### QI-SMOTE: Quantum-Inspired Synthetic Oversampling for Imbalanced Medical Data
- 📅 **2025-09-02** | 🔗 `arxiv.org/abs/2509.02863`
- **Авторы:** Kashtriya V., Singh P.
- **Идея:** Quantum-inspired approach к SMOTE для imbalanced medical data

---

## 📊 Сводка: coverage matrix

| Тема | Статья | Год | Статус в твоём проекте |
|------|--------|-----|------------------------|
| AutoML | AutoML-Med | 2025 | Ручной pipeline |
| AutoML + clinic | CliMB (van der Schaar) | 2024 | — |
| Early prediction | ICU First-Day Review | 2025 | Частично (первые измерения) |
| Multicenter ext. val. | MIMIC×4 Multimodal | 2025 | Пока одна когорта |
| Federated learning | FLICU + OneFlorida+ | 2022-26 | — |
| Causal inference | EHR Causal DL | 2020-21 | Только SHAP (association) |
| Rashomon effect | Intervention Efficiency | 2025 | 5 моделей с близким AUC |
| Bayesian uncertainty | Deep Bayesian GP | 2020-24 | — |
| Model compression | OrthKD | 2026 | — |

---

## 🎯 Приоритетные направления для расширения проекта

| # | Направление | Что даёт | Усилия |
|---|-------------|----------|:------:|
| 1 | **Rashomon analysis** | Формальный критерий выбора модели из 5 | Низкие |
| 2 | **Causal framing** | Отличие association (SHAP) от causation | Средние |
| 3 | **External validation** (MIMIC/eICU) | Основное требование журналов | Средние |
| 4 | **AutoML baseline** | Сравнение ручного pipeline с AutoML | Низкие |
| 5 | **Federated learning** | При масштабировании на несколько клиник | Высокие |
