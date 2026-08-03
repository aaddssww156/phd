# 📚 Обзор литературы: Предсказание кардиогенного шока в ОРИТ методами ML

> Дата поиска: 31.07.2026
> Базы: arXiv, Semantic Scholar
> Проект: side-project/mine — baseline.ipynb

---

## Структура директории

| Файл | Содержание |
|------|------------|
| `00_overview.md` | Этот обзор — навигация и ключевые выводы |
| `01_cardiogenic_shock.md` | Прямые аналоги: CShock, CardShock score |
| `02_icu_ml_methods.md` | ML в ICU: XGBoost, SHAP, MIMIC, CatBoost |
| `03_methodology.md` | Методология: Brier score, дисбаланс классов, MICE |
| `04_time_series_models.md` | Нерегулярные временные ряды: Latent ODE, STAR-Set, Transformers |
| `05_imputation_augmentation.md` | Импутация и аугментация: TDI, missingness, VAE-GMM |
| `06_uncertainty_survival.md` | Uncertainty: conformal prediction, survival analysis |
| `07_fresh_papers_2024_2026.md` | 🆕 Свежие статьи (2024–2026): хронология, топ-5 для немедленного применения |
| `08_additional_angles.md` | Редкие события, fairness, deployment, идентичный техностек |
| `09_multimodal_tabular_ts.md` | 🔗 Мультимодальный fusion: таблицы + временные ряды в кардиологии |
| `10_extended_coverage.md` | AutoML, федеративное обучение, causal inference, early prediction |

---

## Ключевые выводы для проекта

### 🔴 Требует внимания
**Коррекция дисбаланса классов может ухудшать калибровку.** Sirikul et al. (2026) показали на GUSTO-I (n=40 830), что `class_weight='balanced'`, oversampling и другие методы коррекции:
- Не улучшают AUC
- Ухудшают калибровку (Brier score, calibration curves)
- Повышают нестабильность предсказаний

Текущий проект использует `class_weight='balanced'` (RF, LGBM), `scale_pos_weight` (XGBoost), `auto_class_weights='Balanced'` (CatBoost). Рекомендуется ablation study.

### 🟡 Заслуживает обсуждения
**MICE vs детерминированная импутация для clinical prediction.** Mi et al. (2024) аргументируют, что для prediction models детерминированная импутация предпочтительнее: не включает target в модель импутации → легко применять к новым пациентам.

### 🟢 Сильные стороны проекта
1. Систематическое наращивание предикторов в 4 этапа — редкий подход
2. MICE через LightGBM — современный выбор
3. 5 моделей + Optuna — rigorous comparison
4. SHAP для CatBoost — технически нетривиально
5. Train/Calib/Test + Repeated Stratified KFold

### 🟢 Возможные улучшения
1. Сравнение с **CardShock score** как baseline (AUROC ~0.52 в CShock paper)
2. **Decision Curve Analysis** — функция `decision_curve_data()` уже есть
3. Ablation study по балансировке классов
4. Сравнение MICE vs SimpleImputer
5. **Missingness indicators** — добавить бинарные признаки факта пропуска (Fleming 2019, Qian 2024)
6. **Conformal prediction** для uncertainty quantification (Angelopoulos 2021)
7. **Survival analysis** вместо бинарной классификации — время до КШ + competing risks (Chen 2024)

---

## Самые важные статьи (must-read)

| # | Статья | Год | Ключевой вклад |
|---|--------|-----|----------------|
| 1 | CShock: Dynamic Risk Score for Cardiogenic Shock | 2023 | Прямой аналог, DL, AUROC 0.82 |
| 2 | XGBoost ICU Heart Failure MIMIC-III | 2024 | Идентичная методология, AUROC 0.92 |
| 3 | Class Imbalance Corrections Failed... | 2026 | Коррекция дисбаланса вредит калибровке |
| 4 | Weighted Brier Score for Clinical Utility | 2024 | Рамка для оценки clinical utility |
| 5 | MICE vs Deterministic Imputation for Prediction | 2024 | Детерминированная импутация для production |
| 6 | Latent ODEs for Irregularly-Sampled Time Series | 2019 | Neural ODE — фундамент для нерегулярных рядов |
| 7 | Conformal Prediction: A Gentle Introduction | 2021 | Distribution-free uncertainty для любой модели |
| 8 | Deep Survival Analysis (Chen monograph) | 2024 | От Cox до Neural ODE для time-to-event |
| 9 | Beyond Random Missingness: Clinical Rethinking | 2024 | Random masking не отражает клиническую реальность |
| 10 | STAR-Set: Structure-Aware Set Transformers | 2026 | Attention biases для асинхронных EHR |

---

## Глоссарий сокращений

| Сокращение | Расшифровка |
|------------|-------------|
| КШ | Кардиогенный шок |
| ОРИТ / ICU | Отделение реанимации и интенсивной терапии |
| ОСН | Острая сердечная недостаточность |
| ОИМ / MI | Острый инфаркт миокарда |
| СН / HF | Сердечная недостаточность |
| SII | Systemic Immune-Inflammation Index |
| KDIGO | Kidney Disease: Improving Global Outcomes |
| MICE | Multiple Imputation by Chained Equations |
| SHAP | SHapley Additive exPlanations |
| DCA | Decision Curve Analysis |
