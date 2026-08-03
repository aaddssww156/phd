---
title: "Синтез инсайтов из 72+ статей для проекта «Предсказание КШ в ОРИТ»"
date: 2026-07-31
tags: [synthesis, recommendations, cardiogenic-shock, ML, ICU]
---

# 🧠 Синтез: инсайты, рекомендации, направления

> На основе анализа 72+ статей (2017–2026), отобранных по релевантности к проекту.

---

## 🔴 1. КРИТИЧЕСКИЕ МЕТОДОЛОГИЧЕСКИЕ ПРОБЛЕМЫ (fix before submission)

### 1.1 Коррекция дисбаланса классов вредит калибровке

**Источник:** [[class_imbalance_harm]] (Sirikul et al., 2026), [[rare_events_metrics]] (Minus et al., 2025)

**Проблема:** Ты используешь `class_weight='balanced'` (RF, LGBM), `scale_pos_weight` (XGBoost), `auto_class_weights='Balanced'` (CatBoost). Sirikul et al. показали на GUSTO-I (n=40 830), что:
- Коррекция дисбаланса **не улучшает AUC**
- Приводит к **miscalibration** (систематическое завышение риска)
- Повышает **prediction instability**

**Рекомендация:**
> 🔧 Проведи ablation study: обучи все 4 модели без коррекции дисбаланса. Сравни AUC, Brier score, calibration curves. Если Brier улучшится при том же AUC — откажись от class_weight. Это сильный результат для статьи (подтверждает Sirikul 2026 на независимых данных).

### 1.2 AUC может быть misleading при prevalence ~7-8%

**Источник:** [[rare_events_metrics]] (Minus et al., 2025), [[brier_misconceptions]] (Hoessly, 2025)

**Проблема:** При 7-8% positive class, AUC завышает perceived performance.

**Рекомендация:**
> 🔧 Primary metric = **PR-AUC** (уже считаешь — выдвини на первый план). Дополнительно: **Net Benefit** (DCA), **Weighted Brier score** [[weighted_brier_score]] (Zhu, 2024). Не полагайся только на AUC.

### 1.3 MICE хорош, но для production — спорно

**Источник:** [[mice_vs_deterministic_imputation]] (Mi et al., 2024), [[mice_rf_vs_dl_imputation]] (Le et al., 2024)

**Проблема:** MICE требует итеративной подгонки на всех данных, включая target → невозможно применить к одному новому пациенту в production.

**Рекомендация:**
> 🔧 Сравни MICE с SimpleImputer (median) как baseline. Покажи, что MICE даёт прирост, но для production предлагай обученный SimpleImputer с теми же параметрами. Или: используй MICE только для train, а на production — детерминированную импутацию с параметрами, сохранёнными из MICE.

### 1.4 Пропуски — это сигнал

**Источник:** [[missingness_as_stability]] (Fleming et al., 2019), [[beyond_random_missingness]] (Qian et al., 2024)

**Проблема:** Клинические пропуски не случайны (MNAR): SpO2 не измеряют у стабильных, тесты назначают при подозрении на ухудшение.

**Рекомендация:**
> 🔧 Добавь **missingness indicators** — бинарные признаки `{feature}_was_missing`. Дай модели использовать сам факт отсутствия измерения как предиктор. Минимальные усилия → потенциально значимый прирост AUC.

---

## 🟡 2. МОДЕЛИ: ЧТО ВЫБРАТЬ И КАК НАСТРОИТЬ

### 2.1 CatBoost — правильный выбор для клинических таблиц

**Источник:** [[catboost_elderly_icu_dm_hf]] (Fan et al., 2025), [[shap_optuna_xgb_lgbm_catboost]] (de Oliveira et al., 2025)

**Инсайт:** На клинических табличных данных с категориальными признаками CatBoost consistently превосходит XGBoost и LightGBM при прочих равных.

**Рекомендация:**
> 🔧 Сохрани все 5 моделей для comparison table, но основную ставку делай на CatBoost. Подчеркни: native handling of categorical features без one-hot encoding (в отличие от XGBoost/LGBM, требующих ручного кодирования).

### 2.2 Optuna — правильный выбор для HPO

**Источник:** [[shap_optuna_xgb_lgbm_catboost]] (de Oliveira et al., 2025)

**Инсайт:** Optuna (TPE sampler) consistently находит лучшие конфигурации, чем Grid-Search, при том же числе trials. Это подтверждено на идентичном техностеке.

**Рекомендация:**
> 🔧 В статье подчеркни: «We used Optuna with TPE sampler (50 trials) rather than Grid-Search, as TPE has been shown to find better configurations with the same computational budget in clinical prediction tasks.»

### 2.3 TICL/Foundation models не ready для imbalanced clinical data

**Источник:** [[retrieval_tabular_fm_ehr]] (Pham et al., 2026)

**Инсайт:** Tabular in-context learning models (PFN-based) **деградируют** при class imbalance и distribution shift.

**Рекомендация:**
> 🔧 В Discussion: «While tabular foundation models show promise, recent benchmarks demonstrate that classical gradient boosting methods remain more robust under the class imbalance and data constraints typical of clinical prediction tasks (Pham et al., 2026).»

---

## 🟢 3. EVALUATION FRAMEWORK: ЧТО ТРЕБУЮТ РЕЦЕНЗЕНТЫ

### 3.1 Multi-dimensional evaluation — стандарт 2025-2026

**Источник:** [[t2d_external_validation_fairness]] (Pall et al., 2026), [[brier_misconceptions]] (Hoessly, 2025)

**Инсайт:** Рецензенты ожидают evaluation по 4 осям: discrimination + calibration + interpretability + fairness.

**Рекомендация:**
> 🔧 Твой evaluation уже покрывает 3 из 4:
> - ✅ **Discrimination:** ROC-AUC, PR-AUC
> - ✅ **Calibration:** Brier score, calibration curves, reliability diagrams
> - ✅ **Interpretability:** SHAP (summary, bar, dependence, waterfall)
> - ❌ **Fairness:** отсутствует
>
> Добавь стратификацию метрик по: полу, возрасту, классу Killip. Минимум — таблица AUC/Brier для подгрупп.

### 3.2 Decision Curve Analysis — must-have для clinical utility

**Инсайт:** DCA (у тебя уже есть функция `decision_curve_data()`) — это то, что превращает «хорошую модель» в «клинически полезную модель».

**Рекомендация:**
> 🔧 Обязательно включи DCA в основные результаты. Покажи, что модель имеет positive Net Benefit в клинически релевантном диапазоне порогов (10-40% для КШ).

### 3.3 External validation — главное требование журналов

**Источник:** [[multimodal_mortality_multicenter_4dbs]] (Mamandipoor et al., 2025), [[xgboost_mods_elderly_multicenter]] (Liu et al., 2020)

**Инсайт:** Без external validation статью не примут в хороший журнал. Multicenter — золотой стандарт.

**Рекомендация:**
> 🔧 Если нет второй когорты: используй **bootstrap internal validation** как компромисс (Mi et al., 2024). Или: валидируйся на публичном MIMIC-IV/eICU с максимально похожими inclusion criteria. В limitations честно укажи: single-center, нужна external validation.

---

## 🧩 4. FEATURE ENGINEERING: НЕОЧЕВИДНЫЕ ХОДЫ

### 4.1 Missingness indicators — дешёвый прирост

[[missingness_as_stability]] (Fleming, 2019)

Для каждого признака с пропусками добавь бинарный `{feature}_was_measured`. Модель сама выучит, что отсутствие измерения SpO2 у стабильного пациента ≠ отсутствие измерения лактата у тяжёлого.

### 4.2 Статические агрегаты временных рядов

[[static_mts_fusion_amr]] (Martinez-Aguero, 2024), [[multimodal_icu_deterioration_bilstm]] (Sadanandan, 2026)

Если у тебя есть несколько измерений ЧСС/SpO2 во времени — не бери только первое. Добавь:
- `ЧСС_mean`, `ЧСС_std`, `ЧСС_trend` (разность первого и последнего)
- `SpO2_min`, `SpO2_drop` (минимальное значение, падение от baseline)
- `MAP_drop` (max падение среднего АД)

Это даёт модели информацию о динамике без необходимости менять архитектуру.

### 4.3 SHAP-guided feature selection vs экспертное наращивание

[[shap_optuna_xgb_lgbm_catboost]] (de Oliveira, 2025)

Твой подход: 4 этапа экспертного наращивания признаков. Альтернатива: обучить на всех признаках и использовать mean |SHAP| для ранжирования.

**Идея для статьи:**
> 🔧 Сравни два подхода к feature selection в Discussion: (1) clinician-driven staged addition (твой) vs (2) data-driven SHAP-based selection. Покажи, что они сходятся на одном наборе топ-признаков → mutual validation.

---

## 🔬 5. SHAP-ИНТЕРПРЕТАЦИЯ: BEST PRACTICES

### 5.1 SHAP ≠ causation

**Источник:** [[causal_inference_medicine_summary]] (Zhang et al., 2021), [[causal_effects_ehr_dl]] (Li et al., 2020)

**Инсайт:** SHAP показывает **association**, а не **causation**. SII может быть повышен потому, что КШ уже развивается (обратная причинность), а не потому, что высокий SII вызывает КШ.

**Рекомендация:**
> 🔧 В Discussion добавь параграф «Limitations of SHAP»: «SHAP values reflect associative relationships within the trained model and should not be interpreted as causal effects. Prospective intervention studies are needed to establish causality.»

### 5.2 Waterfall plot для клиницистов

Твой `shap_explain()` уже строит waterfall для high-risk пациента — это самое сильное с точки зрения clinical communication. Покажи его в основных результатах, а не в supplement.

### 5.3 ALE plots как альтернатива SHAP dependence

[[catboost_elderly_icu_dm_hf]] (Fan, 2025)

SHAP dependence plots чувствительны к корреляции признаков. ALE (Accumulated Local Effects) plots — робастная альтернатива. Для correlated clinical features (SII коррелирует с лейкоцитами и тромбоцитами) ALE может быть корректнее.

---

## 🚀 6. НАПРАВЛЕНИЯ ДЛЯ РАЗВИТИЯ ПРОЕКТА

### 6.1 Ближайшие (1-2 недели)

| # | Действие | Эффект |
|---|----------|--------|
| 1 | **Ablation: без class_weight** | Потенциально лучше калибровка + сильный result |
| 2 | **Missingness indicators** | Дешёвый прирост AUC |
| 3 | **Fairness audit** (пол, возраст, Killip) | Закрывает 4-ю ось evaluation |
| 4 | **DCA график** в основные результаты | Clinical utility — требование рецензентов |
| 5 | **Сравнение MICE vs SimpleImputer** | Production readiness discussion |

### 6.2 Среднесрочные (месяц+)

| # | Действие | Эффект |
|---|----------|--------|
| 6 | **Conformal prediction** на CatBoost [[conformal_prediction_intro]] | Prediction intervals с гарантиями → clinical deployment |
| 7 | **Survival analysis** вместо бинарной [[deep_survival_analysis_monograph]] | Время до КШ + competing risks → более богатая модель |
| 8 | **External validation** на MIMIC-IV/eICU | Требование журналов |
| 9 | **Rashomon analysis** [[rashomon_effect_clinical]] | Формальный критерий выбора из 5 моделей |

### 6.3 Будущие (PhD trajectory)

| # | Направление | Ключевая статья |
|---|-------------|-----------------|
| 10 | **Multimodal: статика + временные ряды** | [[medpatch_multimodal_fusion]], [[tfn_temporal_fusion_nexus]] |
| 11 | **Federated learning** для multicenter | [[federated_oneflorida_postop]] |
| 12 | **Causal inference** для understanding | [[causal_effects_ehr_dl]] |
| 13 | **Agentic AI** clinical decision support | [[agentic_ai_medicine_review]] |

---

## 📝 7. ЧТО НАПИСАТЬ В СТАТЬЕ: КЛЮЧЕВЫЕ MESSAGES

### Сильные стороны (подтверждённые литературой)

1. **Staged feature addition** (4 этапа) — редкий и methodologically sound подход. Подкрепляется [[static_mts_fusion_amr]], [[xgboost_icu_hf_mimic]].
2. **5-model comparison** (LogReg → CatBoost) — rigorous. Подкрепляется [[shap_optuna_xgb_lgbm_catboost]], [[catboost_elderly_icu_dm_hf]].
3. **MICE через LightGBM** — современный выбор. Подкрепляется [[mice_rf_vs_dl_imputation]] (MICE-RF превосходит DL).
4. **Optuna TPE** эффективнее Grid-Search. Подкрепляется [[shap_optuna_xgb_lgbm_catboost]].
5. **SHAP для CatBoost** — технически нетривиально (твой `_catboost_shap()` fix). Подкрепляется: сложность SHAP для CatBoost известна в community.
6. **Train/Calib/Test split** + Repeated Stratified KFold — rigorous validation.

### Уникальные преимущества перед CShock

[[cshock_dynamic_risk_score]] (Hu et al., 2023) — прямой аналог:

| | CShock | Твой проект |
|---|--------|-------------|
| Метод | Deep Learning | Gradient Boosting (5 моделей) |
| Интерпретация | Нет | SHAP (summary, bar, dependence, waterfall) |
| Калибровка | Нет | Brier score, calibration curves, DCA |
| Признаки | Все доступные | 4 этапа наращивания |
| Импутация | Не указана | MICE (LightGBM) |

> 💡 **Главный message статьи:** «We demonstrate that a rigorous gradient boosting pipeline with staged feature addition, proper calibration, and SHAP interpretability can match or exceed deep learning approaches for cardiogenic shock prediction, while providing clinically actionable explanations.»

---

## ⚠️ 8. ЧЕСТНЫЕ ОГРАНИЧЕНИЯ (не прячь — рецензенты оценят)

1. **Single-center data** — external validation needed ([[multimodal_mortality_multicenter_4dbs]])
2. **Retrospective design** — prospective validation needed
3. **SHAP ≠ causation** — association only ([[causal_inference_medicine_summary]])
4. **Missingness pattern** — MNAR likely, MICE assumes MAR
5. **Rashomon effect** — 5 моделей с близким AUC ([[rashomon_effect_clinical]])
6. **Prevalence ~7-8%** — AUC interpretation caveats ([[rare_events_metrics]])
