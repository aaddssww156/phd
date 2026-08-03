# ⏱️ Нерегулярные временные ряды в клинических данных

> **Контекст:** задача прогнозирования летальности в ОРИТ по коротким гетерогенным временным рядам — классическая проблема ICU time-series modeling.
> Ключевые «боли»: короткие ряды, нерегулярная дискретизация, гетерогенный набор признаков между пациентами.

---

## 1. 🏛️ Latent ODEs for Irregularly-Sampled Time Series (фундаментальная)

**Библиография:**
> Rubanova Y., Chen R.T.Q., Duvenaud D. (2019). Latent ODEs for Irregularly-Sampled Time Series. *arXiv:1907.03907*.

| Параметр | Значение |
|----------|----------|
| Задача | Моделирование нерегулярных временных рядов |
| Ключевая идея | Обобщение RNN → непрерывная динамика через ОДУ (ODE-RNN) |
| Инновация | Latent ODE: ODE-RNN заменяет recognition network |
| Бонус | Моделирует время наблюдений через Poisson process |
| Результат | ODE-based модели превосходят RNN на нерегулярных данных |
| Ссылка | https://arxiv.org/abs/1907.03907 |

### Почему это важно
Это foundational paper, который открыл направление Neural ODE для клинических временных рядов. В отличие от дискретных RNN, ODE-RNN определяет скрытую динамику как непрерывный процесс:

```
dh/dt = f(h, t; θ)
```

Это позволяет естественно обрабатывать произвольные промежутки между наблюдениями — именно то, что нужно для данных ОРИТ, где показатели снимаются с разной частотой.

### Идея для проекта
Если твои данные содержат временные срезы (ЧСС, SpO2, АД снимаются каждые 1–4 часа), Latent ODE может дать более богатое представление, чем агрегация (mean/min/max).

---

## 2. 🕐 Continuous-Time Linear Positional Embedding (CTLPE)

**Библиография:**
> Kim B., Lee J.-G. (2024). Continuous-Time Linear Positional Embedding for Irregular Time Series Forecasting. *arXiv:2409.20092*.

| Параметр | Значение |
|----------|----------|
| Задача | Positional embedding для Transformer на нерегулярных рядах |
| Проблема | Transformer требует позиционного кодирования, стандартные подходы — для равномерных рядов |
| Решение | Непрерывная линейная функция для кодирования времени |
| Сравнение | Neural CDE-based positional embedding уступает линейной |
| Ссылка | https://arxiv.org/abs/2409.20092 |

### Ключевая формула
Вместо дискретного sin/cos позиционного кодирования, CTLPE использует непрерывную функцию:

```
PE(t) = W · t + b
```

где `t` — реальное время события (timestamp), а не номер позиции в последовательности.

### Практический смысл
Если ты решишь добавить временную компоненту (например, динамику ЧСС и SpO2 за первые 6 часов), Transformer с CTLPE — это state-of-the-art способ учесть нерегулярность измерений.

---

## 3. 🏥 STAR-Set: Structure-Aware Set Transformers для EHR

**Библиография:**
> Lee J., Lee K., Kim C., Yang E. (2026). Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series. *arXiv:2603.06605*.

| Параметр | Значение |
|----------|----------|
| Задача | Классификация асинхронных клинических временных рядов (EHR) |
| Идея | Attention biases: temporal locality + variable-type affinity |
| Temporal bias | `-|Δt|/τ` с обучаемым масштабом τ |
| Variable bias | Матрица совместимости признаков B |
| Результаты (ICU) | CPR: 0.716, Mortality: 0.916, Vasopressor: 0.837 (AUC) |
| Ссылка | https://arxiv.org/abs/2603.06605 |

### Почему это важно для тебя
Эта работа прямо адресует проблему EHR как **асинхронных многомерных временных рядов**. Ключевой инсайт:

- **Grid-подход** (равномерная сетка) требует импутации пропусков → bias
- **Set-подход** (токенизация событий) теряет временную структуру
- **STAR-Set** — средний путь: point-set токенизация + soft attention biases

Результат: AUC 0.916 на mortality prediction — это сильный baseline.

---

## 4. 🔗 Multimodal Deep Learning: ICU Deterioration (MDS-ICU)

**Библиография:**
> López Alcaraz J.M. et al. (2026). A Multimodal Deep Learning Framework for Predicting ICU Deterioration: Integrating ECG Waveforms with Clinical Data. *arXiv:2601.06645*.

| Параметр | Значение |
|----------|----------|
| Задача | 33 клинических исхода в ICU (multi-task) |
| Данные | MIMIC-IV: 63 001 сэмплов, 27 062 пациента |
| Модальности | Демография + витальные + лаборатория + **ECG** + процедуры |
| Архитектура | S4 (state space) для ECG + RealMLP для табличных данных |
| AUROC | 24h mortality: **0.90**, ИВЛ: **0.97**, седация: 0.92 |
| Калибровка | Отличное совпадение predicted vs observed |
| Бенчмарк | Модель > клиницисты > LLM |
| Ссылка | https://arxiv.org/abs/2601.06645 |

### Ключевой вывод
**Multimodal fusion работает.** Даже без ECG, подход «структурированные данные + временные ряды» показывает AUROC 0.90 на mortality. Это релевантно, если у тебя есть временные срезы показателей.

### Что можно позаимствовать
- RealMLP — современная MLP-архитектура для табличных данных (альтернатива CatBoost/XGBoost)
- Multi-task learning: одна модель предсказывает 33 исхода → лучше generalizability

---

## 5. 📝 Physiological Time Series + Clinical Notes (fusion)

**Библиография:**
> Shukla S.N., Marlin B.M. (2020). Integrating Physiological Time Series and Clinical Notes with Deep Learning for Improved ICU Mortality Prediction. *arXiv:2003.11059*.

| Параметр | Значение |
|----------|----------|
| Задача | Mortality prediction в ICU: временные ряды + текст |
| Архитектура | Interpolation-prediction network |
| Fusion | Early vs late fusion модальностей |
| Результат | Late fusion даёт статистически значимое улучшение |
| Ссылка | https://arxiv.org/abs/2003.11059 |

---

## 6. 🧱 EHR-RAGp: Retrieval-Augmented Foundation Model для EHR

**Библиография:**
> Shurrab S., Al-Omari M., El Samad D., Shamout F.E. (2026). EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for EHR. *arXiv:2605.12335*.

| Параметр | Значение |
|----------|----------|
| Идея | RAG для EHR: извлекать релевантную историю пациента динамически |
| Инновация | Prototype-guided retrieval module — alignment mechanism |
| Результат | Превосходит SOTA EHR foundation models и transformer baselines |
| Ссылка | https://arxiv.org/abs/2605.12335 |

### Почему это интересно
Это cutting-edge (май 2026): foundation model + retrieval augmentation для EHR. Идея в том, что вместо фиксированного окна истории, модель динамически извлекает релевантные чанки patient history. Для твоей задачи это может значить: модель сама решает, какие эпизоды истории пациента важны для предсказания КШ.

---

## 📊 Карта методов для нерегулярных временных рядов

```
Подходы к нерегулярным временным рядам в EHR:

1. ДИСКРЕТИЗАЦИЯ + ИМПУТАЦИЯ
   ├── Равномерная сетка + LOCF (просто, но bias)
   ├── Равномерная сетка + MICE (твой текущий подход — для статики)
   └── Равномерная сетка + GRU-D / BRITS (учитывают missingness)

2. НЕПРЕРЫВНОЕ ПРЕДСТАВЛЕНИЕ
   ├── Latent ODE (Rubanova 2019) — ODE-RNN
   ├── Neural CDE (Kidger 2020) — контролируемые дифф. уравнения
   └── CTLPE (Kim 2024) — непрерывное позиционное кодирование

3. SET-ФУНКЦИИ (события как множество)
   ├── Deep Sets (Zaheer 2017) — базовый set-подход
   ├── SeFT (Horn 2020) — set functions for time series
   └── STAR-Set (Lee 2026) — attention biases для structure-aware

4. FOUNDATION MODELS
   └── EHR-RAGp (Shurrab 2026) — retrieval-augmented для длинной истории
```

---

## Практические рекомендации для проекта

| # | Идея | Сложность | Потенциал |
|---|------|:---------:|:---------:|
| 1 | Учёт missingness pattern как отдельного признака (не только импутация) | Низкая | Средний |
| 2 | Добавить временные агрегаты (mean/min/max/trend за первые N часов) | Низкая | Высокий |
| 3 | GRU-D / BRITS вместо MICE для временных рядов | Средняя | Средний |
| 4 | STAR-Set для асинхронных событий в EHR | Высокая | Высокий |
| 5 | Multimodal fusion: статика (Killip, ХОБЛ) + динамика (ЧСС, SpO2) | Средняя | Высокий |
