---
title: "Временные ряды + табличные данные: практический roadmap"
date: 2026-07-31
tags: [time-series, tabular, fusion, roadmap, cardiogenic-shock]
---

# ⏱️→🔗 Временные ряды + таблицы: практический roadmap

> На основе анализа 72+ статей. Три сценария — от минимальных изменений до production-grade multimodal системы.

---

## 📋 Исходные данные (моё понимание)

| Тип | Примеры | Частота | Пропуски |
|-----|---------|---------|----------|
| **Табличные (статика)** | Killip, ХОБЛ, TIMI, возраст, пол | Однократно | Умеренные |
| **Лабораторные (статика)** | Глюкоза, SII, Apache II, KDIGO | Однократно (при поступлении) | Есть |
| **Витальные (динамика)** | ЧСС, SpO2, АД сист/диаст | Каждые 15-60 мин | Значительные (MNAR) |

**Ключевая особенность:** измерения нерегулярны и асинхронны — ЧСС каждые 15 мин, SpO2 каждый час, АД при измерении. Это **асинхронные многомерные временные ряды с информативными пропусками**.

---

## 🎯 Сценарий A: «Feature Engineering» (≈2 дня)

> **Суть:** не меняем архитектуру (остаёмся на CatBoost/XGBoost), добавляем статистики временных рядов как новые табличные признаки.

### Что делаем

Для каждого витального показателя за первые N часов (например, 6, 12, 24) вычисляем:

```
Для ЧСС(t):
  ├── ЧСС_first        — первое измерение (уже есть)
  ├── ЧСС_mean         — среднее за N часов
  ├── ЧСС_std          — вариабельность (новый признак!)
  ├── ЧСС_min, ЧСС_max — экстремумы
  ├── ЧСС_trend        — (последнее − первое) / время
  ├── ЧСС_slope        — коэффициент линейной регрессии
  └── ЧСС_was_measured — доля времени с измерениями (missingness indicator)

Для SpO2(t):
  ├── SpO2_min         — минимальная сатурация (клинически важно!)
  ├── SpO2_drop        — падение от baseline
  ├── SpO2_time_below_90% — время гипоксемии
  └── SpO2_was_measured

Для АД(t):
  ├── MAP_mean         — среднее артериальное давление
  ├── MAP_drop         — максимальное падение
  ├── Shock_Index      — ЧСС / АД_сист (если ещё не считали)
  └── Pulse_Pressure   — АД_сист − АД_диаст
```

### Плюсы
- **Ноль изменений в архитектуре** — те же CatBoost/XGBoost
- **Интерпретируемость** — SHAP работает как раньше
- **Быстро** — только feature engineering
- **Статья:** можно показать, что агрегаты временных рядов улучшают AUC по сравнению с одним первым измерением

### Минусы
- Теряется тонкая временная структура (форму волны)
- Не используется cross-variable взаимодействие во времени (ЧСС падает → SpO2 падает через 30 мин)
- Выбор окна агрегации (6h, 12h, 24h) субъективен

### Статьи в поддержку
- [[missingness_as_stability]] — missingness indicators
- [[static_mts_fusion_amr]] — static + MTS fusion paradigm
- [[multimodal_icu_deterioration_bilstm]] — show aggregates work

---

## 🎯 Сценарий B: «Late Fusion» (≈1-2 недели)

> **Суть:** отдельные энкодеры для статики и динамики → слияние на последнем слое. Меняем архитектуру, но остаёмся в рамках gradient boosting для статики.

### Архитектура

```
                    ┌─────────────────────┐
[Killip, ХОБЛ, ...] │  CatBoost / XGBoost │──┐
(таблицы)           │  (как сейчас)       │  │
                    └─────────────────────┘  │
                                             ├──→ [Fusion Layer] → P(КШ)
                    ┌─────────────────────┐  │      (MLP или
[ЧСС(t), SpO2(t),   │  Временной энкодер  │──┘       weighted sum)
 АД(t)]             │  (GRU / LSTM /      │
(временные ряды)    │   Transformer)      │
                    └─────────────────────┘
```

### Варианты временного энкодера

| Энкодер | Сложность | Плюсы | Минусы | Статья |
|---------|:---------:|-------|--------|--------|
| **GRU-D** | Низкая | Учитывает missingness pattern, хорошо для коротких рядов | Устарел концептуально | — |
| **LSTM + mean pooling** | Низкая | Просто, работает | Теряет temporal resolution | [[multimodal_icu_deterioration_bilstm]] |
| **Transformer + CTLPE** | Средняя | SOTA positional encoding для нерегулярных рядов | Больше параметров | [[ctlpe_irregular_ts]] |
| **STAR-Set** | Высокая | Attention biases для асинхронных EHR, лучший AUC | Сложен в имплементации | [[star_set_async_ehr]] |

### Рекомендованный выбор: **GRU + CTLPE positional encoding**

**Почему:**
1. У тебя **короткие ряды** (первые 24 часа, ~24-96 измерений на показатель) — Transformer overkill, GRU/LSTM достаточно
2. CTLPE даёт непрерывное позиционное кодирование для нерегулярных интервалов
3. GRU-D встроенно обрабатывает missingness (в отличие от LSTM)
4. Проще обучить на малой выборке ( Transformer требует больше данных)

### Что меняется в пайплайне

```python
# Сейчас:
X_static = [Killip, HR_first, SpO2_first, ...]  # (n_patients, n_features)
model = CatBoostClassifier()
model.fit(X_static, y)

# Сценарий B:
X_static  = [Killip, ХОБЛ, TIMI, glucose, SII, Apache, KDIGO]
X_temporal = [HR(t), SpO2(t), MAP(t)]  # (n_patients, n_timesteps, n_channels)

static_encoder  = CatBoostClassifier(...)        # остаётся как есть
temporal_encoder = GRU( hidden=64, ...)          # новый компонент

static_emb   = static_encoder.transform(X_static)   # или predict_proba
temporal_emb = temporal_encoder(X_temporal)[:, -1]  # последнее скрытое состояние

fusion_input = concat([static_emb, temporal_emb])
fusion_head  = LogisticRegression()  # или MLP
prediction   = fusion_head.fit(fusion_input, y)
```

### Плюсы
- **Каждый энкодер специализируется** на своём типе данных
- **Late fusion** статистически значимо лучше early fusion для клиники — [[ts_clinical_notes_fusion]]
- **Интерпретируемость:** SHAP для статики + attention weights / integrated gradients для динамики
- Можно предобучить энкодеры раздельно

### Минусы
- Два отдельных обучения → два набора гиперпараметров
- Нет cross-modal взаимодействия (ЧСС влияет на важность Killip — это теряется)
- GRU может быть избыточен, если временная динамика слабая

### Статьи в поддержку
- [[ts_clinical_notes_fusion]] — late fusion > early fusion (Shukla & Marlin, 2020)
- [[ctlpe_irregular_ts]] — CTLPE для нерегулярных рядов (Kim & Lee, 2024)
- [[static_mts_fusion_amr]] — static + MTS fusion framework (Martinez-Aguero, 2024)
- [[multimodal_icu_deterioration_bilstm]] — BiLSTM энкодер для ICU (Sadanandan, 2026)

---

## 🎯 Сценарий C: «MedPatch-inspired» (≈2-4 недели)

> **Суть:** production-grade multimodal система с joint+late fusion, confidence-weighted модальностями и корректной обработкой пропусков целых модальностей.

### Архитектура (упрощённый MedPatch)

```
                         ┌──────────────────┐
[Killip, ХОБЛ, TIMI] →  │ Static Encoder    │──┐
                         │ (MLP/CatBoost)    │  │
                         └──────────────────┘  │
                                               ├─→ Joint Fusion ─┐
                         ┌──────────────────┐  │                  │
[ЧСС(t), SpO2(t)]     → │ Temporal Encoder  │──┘                  │
                         │ (GRU + CTLPE)     │                     │
                         └──────────────────┘                     │
                                                                  ├─→ Confidence
                         ┌──────────────────┐                     │    Weighted
[Missingness Detector] → │ Modality Detector │────────────────────┤    Prediction
                         │ (качество/полнота)│                     │
                         └──────────────────┘                     │
                                                                  │
                         ┌──────────────────┐                     │
Static Emb ────────────→ │ Late Fusion Head │────────────────────┘
Temporal Emb ──────────→ │ (MLP)            │
                         └──────────────────┘
```

### Ключевые компоненты

**1. Missingness-aware modality gating**

Пациенты делятся на три группы:
- **Полные данные:** статика + временные ряды → full model
- **Только статика** (короткое пребывание, нет repeated измерений) → static-only branch
- **Только временные ряды** (нет лабораторных при поступлении) → temporal-only branch

```python
if n_temporal_measurements < 3:    # менее 3 измерений за 24h
    prediction = static_model(X_static)
elif lab_data_completeness < 0.5:  # менее половины labs
    prediction = temporal_model(X_temporal)
else:
    prediction = fusion_model(X_static, X_temporal)
```

**2. Confidence-weighted prediction**

Вместо жёсткого выбора ветки — мягкое взвешивание:

```python
conf_static  = sigmoid( quality_score_static  )  # 0..1
conf_temporal = sigmoid( quality_score_temporal )  # 0..1
w_static  = conf_static  / (conf_static + conf_temporal)
w_temporal = conf_temporal / (conf_static + conf_temporal)
prediction = w_static * p_static + w_temporal * p_temporal
```

**3. Cross-modal attention (опционально)**

Если нужен fine-grained cross-modal interaction:

```python
# Q из статики, K,V из временных рядов
cross_attn = Attention(Q=static_emb, K=temporal_seq, V=temporal_seq)
enhanced_static = static_emb + cross_attn
```

### Плюсы
- **Robust to missing modalities** — ключевое для реальной клиники
- **Confidence calibration** — модель знает, когда она неуверенна
- **Самый высокий AUC** — +5-10% над unimodal согласно [[tfn_temporal_fusion_nexus]]
- Можно деплоить в клинику: падает одна модальность → модель продолжает работать

### Минусы
- Сложность имплементации и отладки
- Нужно больше данных для обучения всех веток
- Требует separate validation на каждом subgroup

### Статьи в поддержку
- [[medpatch_multimodal_fusion]] — сама архитектура (Al Jorf & Shamout, 2025)
- [[tfn_temporal_fusion_nexus]] — +10% AUC от multimodal (Kumar et al., 2026)
- [[mind_knowledge_distillation]] — KD для сжатия (Guerra-Manzanares, 2025)
- [[trace_multimodal_ts_fm]] — conditional estimation missing modalities (Kan, 2026)

---

## 📊 Сравнительная таблица сценариев

| Критерий | A: Feature Eng. | B: Late Fusion | C: MedPatch |
|----------|:---:|:---:|:---:|
| **Время реализации** | 2 дня | 1-2 недели | 2-4 недели |
| **Изменение архитектуры** | Нет | Умеренное | Значительное |
| **Ожидаемый прирост AUC** | +1-3% | +3-7% | +5-10% |
| **Интерпретируемость** | Полная (SHAP) | Высокая (SHAP + attention) | Средняя |
| **Robustness к пропускам** | Средняя (импутация) | Средняя (GRU-D) | Высокая (modality gating) |
| **Production readiness** | Высокая (тот же CatBoost) | Средняя (два инференса) | Высокая (graceful degradation) |
| **Публикационный потенциал** | Умеренный | Высокий | Очень высокий |
| **Риск переобучения** | Низкий | Средний | Высокий |

---

## 🎯 Рекомендованный путь

### Для текущей статьи: A → B (последовательно)

1. **Сделай A сейчас** — за 2 дня добавишь агрегаты временных рядов. Покажешь, что даже простые статистики улучшают AUC. Это уже differentiation от CShock (который использует все features сразу, без выделения временной структуры).

2. **B — как отдельная секция/эксперимент** — сравни Late Fusion с Feature Engineering baseline. Покажи, что GRU+CTLPE энкодер даёт дополнительный прирост.

3. **Обсуди C в Discussion** — «Future work: MedPatch-inspired architecture with modality gating for real-world deployment where some modalities may be unavailable.»

### Для PhD в целом: A → B → C

```
Глава 1 (сейчас):     A — статические данные + агрегаты TS
Глава 2 (через 3 мес): B — Late Fusion (GRU + CTLPE + CatBoost)
Глава 3 (через 6 мес): C — MedPatch с modality gating + conformal prediction
```

---

## 🛠️ Конкретный код для Сценария A

```python
def engineer_temporal_features(df, vital_col, time_col, windows=[6, 12, 24]):
    """
    df: long-format DataFrame со столбцами [patient_id, time_hours, HR, SpO2, MAP]
    vital_col: имя колонки витального показателя
    time_col: имя колонки времени
    windows: окна агрегации в часах
    """
    features = {}
    for w in windows:
        mask = df[time_col] <= w
        window_data = df[mask].groupby('patient_id')[vital_col]
        
        features[f'{vital_col}_mean_{w}h']   = window_data.mean()
        features[f'{vital_col}_std_{w}h']    = window_data.std()
        features[f'{vital_col}_min_{w}h']    = window_data.min()
        features[f'{vital_col}_max_{w}h']    = window_data.max()
        features[f'{vital_col}_first_{w}h']  = window_data.first()
        features[f'{vital_col}_last_{w}h']   = window_data.last()
        features[f'{vital_col}_trend_{w}h']  = (
            window_data.last() - window_data.first()
        ) / w
        features[f'{vital_col}_n_meas_{w}h'] = window_data.count()
        
    return pd.DataFrame(features)

# Применяем ко всем витальным показателям
hr_features  = engineer_temporal_features(df_vitals, 'HR', 'hours')
spo2_features = engineer_temporal_features(df_vitals, 'SpO2', 'hours')
map_features  = engineer_temporal_features(df_vitals, 'MAP', 'hours')

# Объединяем со статическими признаками
X_enhanced = X_static.join(hr_features).join(spo2_features).join(map_features)
```

---

## 🛠️ Архитектурный скетч для Сценария B

```python
import torch
import torch.nn as nn

class CTLPE(nn.Module):
    """Continuous-Time Linear Positional Embedding (Kim & Lee, 2024)"""
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, t):
        # t: (batch, seq_len) — реальное время в часах
        return t.unsqueeze(-1) * self.W + self.b  # (batch, seq_len, d_model)

class TemporalEncoder(nn.Module):
    def __init__(self, n_channels, d_model=64, n_layers=2):
        super().__init__()
        self.ctlpe = CTLPE(d_model)
        self.input_proj = nn.Linear(n_channels, d_model)
        self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=True)
        
    def forward(self, x, t, mask):
        # x: (batch, seq_len, n_channels) — витальные показатели
        # t: (batch, seq_len) — время измерений
        # mask: (batch, seq_len) — 1 если измерение было, 0 если пропуск
        
        # Проекция + позиционное кодирование
        x = self.input_proj(x) + self.ctlpe(t)
        
        # GRU с missingness-aware инициализацией
        # (можно заменить на GRU-D для полноценной обработки пропусков)
        x = x * mask.unsqueeze(-1)  # зануляем пропуски
        out, h_n = self.gru(x)
        
        return h_n[-1]  # последнее скрытое состояние

class LateFusionModel(nn.Module):
    def __init__(self, n_static, n_temporal_channels, d_model=64):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, d_model)
        )
        self.temporal_encoder = TemporalEncoder(n_temporal_channels, d_model)
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_static, x_temporal, t, mask):
        emb_static = self.static_encoder(x_static)
        emb_temporal = self.temporal_encoder(x_temporal, t, mask)
        fused = torch.cat([emb_static, emb_temporal], dim=-1)
        return self.fusion_head(fused)
```
