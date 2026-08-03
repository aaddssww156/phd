# 🎯 Uncertainty Quantification и Survival Analysis

---

## 1. 📐 Conformal Prediction: A Gentle Introduction

**Библиография:**
> Angelopoulos A.N., Bates S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *arXiv:2107.07511*.

| Параметр | Значение |
|----------|----------|
| Тип | Hands-on tutorial |
| Идея | Distribution-free uncertainty sets с теоретическими гарантиями |
| Гарантия | Покрытие истинного значения с вероятностью (1-α) |
| Применимость | Любая pre-trained модель (включая CatBoost/XGBoost) |
| Бонус | Код на Python, Jupyter notebooks |
| Ссылка | https://arxiv.org/abs/2107.07511 |

### Что такое conformal prediction (простыми словами)

Вместо точечного предсказания «вероятность КШ = 0.34», conformal prediction выдаёт:

> «С вероятностью 90% истинный риск КШ находится в интервале [0.21, 0.47]»

Это **намного полезнее** для врача, чем одно число. Врач видит неопределённость и принимает решение с её учётом.

### Как это работает

1. Берём обученную модель (твой CatBoost)
2. Вычисляем «nonconformity scores» на calibration set
3. Для нового пациента: предсказание → prediction interval с гарантированным покрытием

### Почему это идеально для твоего проекта

- **Не требует переобучения** — работает с любой готовой моделью
- **Distribution-free** — не нужно предполагать нормальность
- **Finite-sample гарантии** — в отличие от асимптотических доверительных интервалов
- **Прямо в клинику:** «Риск КШ: 22–49% (90% confidence)» → actionable

---

## 2. 🔗 Self-Calibrating Conformal Prediction

**Библиография:**
> van der Laan L., Alaa A.M. (2024). Self-Calibrating Conformal Prediction. *arXiv:2402.07307*.

| Параметр | Значение |
|----------|----------|
| Идея | Объединяет Venn-Abers calibration + conformal prediction |
| Результат | Calibrated point predictions + conditionally valid prediction intervals |
| Расширение | С бинарной классификации на регрессию |
| Ссылка | https://arxiv.org/abs/2402.07307 |

### При чём здесь твой проект

Ты уже делаешь калибровку (calibration curves, Brier score, `CalibratedClassifierCV`). Self-Calibrating Conformal Prediction — это следующий шаг:

- **Обычный подход (твой):** Platt/Isotonic calibration → точечная вероятность → calibration curve
- **SCP:** Venn-Abers calibration → calibration curve **+** prediction intervals с гарантиями

---

## 3. ⏳ Deep Survival Analysis (time-to-event)

**Библиография:**
> Chen G.H. (2024). An Introduction to Deep Survival Analysis Models for Predicting Time-to-Event Outcomes. *arXiv:2410.01086*.

| Параметр | Значение |
|----------|----------|
| Тип | Монография (self-contained introduction) |
| Охват | От Cox proportional hazards до Neural ODE для survival |
| Темы | Competing risks, dynamic prediction, time-varying covariates |
| Бонус | Код для всех моделей и метрик |
| Ссылка | https://arxiv.org/abs/2410.01086 |

### Survival analysis vs бинарная классификация (твой подход)

Сейчас ты предсказываешь **бинарный исход**: разовьётся ли КШ в реанимации (да/нет).

Survival analysis предсказывает **время до события** + **вероятность события в каждый момент времени**:

```
Твой подход:         P(КШ разовьётся когда-либо) = 0.34
Survival analysis:   P(КШ в первые 6 часов)  = 0.05
                     P(КШ в первые 24 часа)  = 0.18
                     P(КШ в первые 72 часа)  = 0.34
                     P(КШ в первые 7 дней)   = 0.41
```

### Что это даёт клинически

- Врач видит, **когда** ждать ухудшения → планирует мониторинг
- Модель учитывает **цензурирование** (пациента выписали до развития КШ)
- **Competing risks:** смерть от другой причины vs КШ
- **Dynamic prediction:** обновление прогноза с каждым новым измерением

---

## 4. 🩺 Deep Survival + Competing Risks

**Библиография:**
> Nemchenko A., Kyono T., van der Schaar M. (2018). Siamese Survival Analysis with Competing Risks. *arXiv:1807.05935*.

| Параметр | Значение |
|----------|----------|
| Задача | Survival analysis с конкурирующими рисками |
| Идея | Siamese networks для разных типов событий |
| Ссылка | https://arxiv.org/abs/1807.05935 |

### Competing risks в контексте КШ

В реанимации у пациента могут быть разные исходы:
1. Развился КШ (твой target)
2. Умер от другой причины (конкурирующий риск)
3. Выписан/переведён (цензурирование)

Классическая бинарная классификация игнорирует конкурирующие риски → смещённые оценки. Survival analysis с competing risks решает эту проблему.

---

## 5. 📊 Weighted Brier Score (clinical utility)

*Подробно в `03_methodology.md`*

**Библиография:**
> Zhu K., Zheng Y., Chan K.C.G. (2024). Weighted Brier Score — an Overall Summary Measure for Risk Prediction Models with Clinical Utility Consideration. *arXiv:2408.01626*.

### Связь с conformal prediction

Weighted Brier score и conformal prediction решают одну задачу с разных сторон:

| | Weighted Brier Score | Conformal Prediction |
|---|---|---|
| **Что даёт** | Скалярная метрика с учётом clinical utility | Prediction intervals с гарантиями |
| **Тип** | Оценка модели | Применение модели |
| **Для кого** | Исследователь (сравнение моделей) | Врач (принятие решения) |

Идеальный пайплайн: Weighted Brier для выбора лучшей модели → Conformal prediction для clinical deployment.

---

## 6. 📈 JANET: Конформное предсказание для временных рядов

**Библиография:**
> English E., Wong-Toi E., Fontana M. et al. (2024). JANET: Joint Adaptive PredictioN-region Estimation for Time-series. *arXiv:2407.06390*.

| Параметр | Значение |
|----------|----------|
| Проблема | Обычный conformal prediction требует exchangeability → не работает для временных рядов |
| Решение | Адаптивные prediction regions для временных рядов |
| Ссылка | https://arxiv.org/abs/2407.06390 |

### Актуально для твоего проекта

Если добавишь временную компоненту (динамику показателей), JANET даёт способ строить prediction intervals с учётом временной зависимости — стандартный conformal prediction на временных рядах невалиден.

---

## 📊 Дорожная карта: от бинарной классификации к clinical deployment

```
СЕЙЧАС:
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ MICE     │ →  │ 5 моделей│ →  │ P(КШ)    │
  │ импутация│    │ + Optuna │    │ = 0.34   │
  └──────────┘    └──────────┘    └──────────┘
  │ Статические признаки (Killip, ЧСС, SpO2, ...)

ШАГ 1: КАЛИБРОВКА
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ MICE     │ →  │ 5 моделей│ →  │ Калибров-│ →  │ P(КШ)    │
  │ импутация│    │ + Optuna │    │ ка (Iso) │    │ = 0.34   │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
  │ Уже делаешь CalibratedClassifierCV + Brier score

ШАГ 2: CONFORMAL PREDICTION
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │ MICE     │ →  │ 5 моделей│ →  │ Conformal│ →  │ P(КШ) ∈      │
  │ импутация│    │ + Optuna │    │ predict  │    │ [0.22, 0.47] │
  └──────────┘    └──────────┘    └──────────┘    └──────────────┘
  │ Самое практичное улучшение (Angelopoulos 2021)

ШАГ 3: SURVIVAL ANALYSIS
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │ MICE     │ →  │ DeepSurv │ →  │ Survival │ →  │ P(КШ ≤ 24h)  │
  │ импутация│    │ / CoxPH  │    │ curve    │    │ = 0.18       │
  └──────────┘    └──────────┘    └──────────┘    └──────────────┘
  │ Время до КШ + competing risks (Chen 2024)

ШАГ 4: MULTIMODAL + ВРЕМЕННЫЕ РЯДЫ
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │ Статика  │ →  │ Fusion   │ →  │ Calib +  │ →  │ P + интервал │
  │ + TDI    │    │ (STAR)   │    │ Conformal│    │ + время      │
  └──────────┘    └──────────┘    └──────────┘    └──────────────┘
  │ Полноценный clinical decision support system
```

---

## Рекомендуемые действия (в порядке практической реализуемости)

| # | Действие | Статья | Усилия | Отдача |
|---|----------|--------|:------:|:------:|
| 1 | Добавить **missingness indicators** | Fleming 2019, Qian 2024 | Минимум | Высокая |
| 2 | **Conformal prediction** на CatBoost | Angelopoulos 2021 | Средние | Очень высокая |
| 3 | **Weighted Brier score** | Zhu 2024 | Низкие | Средняя |
| 4 | **Survival analysis** вместо бинарной | Chen 2024 | Высокие | Очень высокая |
| 5 | **CTLPE + Transformer** для временных срезов | Kim 2024 | Очень высокие | Высокая |
