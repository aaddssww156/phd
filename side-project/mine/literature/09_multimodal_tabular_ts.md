# 🔗 Мультимодальные задачи: табличные данные + временные ряды

> Фокус: слияние (fusion) статических табличных признаков и динамических временных рядов в клинических prediction задачах.
> Контекст: твой проект — табличные предикторы (Killip, ХОБЛ, SII, Apache II) + потенциал добавления временных рядов (ЧСС, SpO2, АД в динамике).

---

## 1. 🏗️ Архитектуры слияния (fusion architectures)

### MedPatch: Confidence-Guided Multi-Stage Fusion for Multimodal Clinical Data
- 📅 **2025-08-07** | 🔗 `arxiv.org/abs/2508.09182`
- **Авторы:** Al Jorf B., Shamout F. (NYU Abu Dhabi)
- **Проблема:** Медицинские мультимодальные данные гетерогенны, малы по размеру, с пропусками модальностей
- **Решение — MedPatch (3 компонента):**
  1. **Multi-stage fusion** — joint + late fusion одновременно
  2. **Missingness-aware** модуль — работает при отсутствии отдельных модальностей
  3. **Confidence-guided patching** — динамический вес модальностей по их уверенности
- **Инсайт:** «Clinical decision-making relies on integration of clinical time-series, medical images, and textual reports»

> 🟢 **Самый релевантный architectural template.** MedPatch решает ровно твою задачу: как совместить табличные данные (Класс Killip, ХОБЛ), временные ряды (ЧСС, SpO2), и при этом корректно обрабатывать ситуацию, когда каких-то измерений нет.

### MIND: Modality-Informed Knowledge Distillation for Multimodal Clinical Prediction
- 📅 **2025-02-03** | 🔗 `arxiv.org/abs/2502.01158`
- **Авторы:** Guerra-Manzanares A., Shamout F.E. (та же группа)
- **Проблема:** Мультимодальные медицинские датасеты **меньше**, чем унимодальные → мультимодальные сети переобучаются
- **Решение:** Knowledge distillation — крупная мультимодальная teacher → компактная unimodal student
- **Инсайт:** «Increase in number of modalities is often associated with an increase in multimodal network size, which may be undesirable in medical use cases»

> 🟢 **Важно для практики:** если добавишь временные ряды к табличным данным, модель станет больше → риск переобучения на малой выборке. MIND предлагает дистилляцию как решение.

---

## 2. 🏥 ICU-Specific Multimodal

### Multimodal DL for Early Prediction of Patient Deterioration in ICU (Time-Series + Clinical Notes)
- 📅 **2026-03-16** | 🔗 `arxiv.org/abs/2603.14719`
- **Автор:** Sadanandan B.
- **Данные:** MIMIC-IV: 74 822 ICU stays → 5.7 million hourly prediction samples
- **Модальности:** Structured time-series (vital signs + labs) + unstructured clinical notes
- **Архитектура:** **BiLSTM** для временных рядов + **ClinicalBERT** для текста → late fusion
- **Задача:** Deterioration (mortality, vasopressor, mechanical ventilation) в течение 24 часов

> 🟢 **Масштаб:** 5.7 млн сэмплов, BiLSTM + трансформер — production-grade подход к ICU multimodal prediction.

### MDS-ICU: Multimodal ICU Deterioration (ECG + Tabular)
- 📅 **2026-01-10** | 🔗 `arxiv.org/abs/2601.06645` *(подробно в `04_time_series_models.md`)*
- **Модальности:** ECG waveforms (S4 state-space encoder) + demographics + vitals + labs (RealMLP)
- **Результат:** AUROC 0.90 (24h mortality), 0.93 (coagulation dysfunction)

### Shukla & Marlin: Physiological Time Series + Clinical Notes
- 📅 **2020-03-24** | 🔗 `arxiv.org/abs/2003.11059` *(подробно в `04_time_series_models.md`)*
- **Сравнение:** Early fusion vs late fusion временных рядов и клинического текста
- **Вывод:** Late fusion статистически значимо лучше унимодальных подходов

---

## 3. 📊 Static + Time Series Fusion (твой будущий сценарий)

### Multimodal Interpretable Models for Antimicrobial Resistance (Static + MTS)
- 📅 **2024-02-09** | 🔗 `arxiv.org/abs/2402.06295`
- **Авторы:** Martínez-Agüero S., Marques A.G., Mora-Jiménez I. et al.
- **Модальности:** **Static data** (демография, коморбидности) + **MTS** (Multivariate Time Series — витальные показатели)
- **Ключевой framework:** Интерпретируемая мультимодальная DNN для clinical prediction
- **Инсайт:** «EHR is an inherently multimodal register characterized by static data and multivariate time series»

> 🟢 **Прямо про твой сценарий:** статические признаки (Killip, ХОБЛ, TIMI) + MTS (ЧСС, SpO2, АД в динамике). Эта статья даёт методологический каркас для сочетания этих модальностей с сохранением интерпретируемости.

### Temporal Fusion Nexus (TFN): Irregular Time Series + Clinical Narratives
- 📅 **2026-01-13** | 🔗 `arxiv.org/abs/2601.08503`
- **Авторы:** Kumar A., Rauch S., Cypko M. et al. (Charité Berlin)
- **Задача:** Post-kidney transplant outcomes (graft loss, rejection, mortality)
- **Данные:** 3 382 пациента
- **Результат:** TFN (multimodal) превосходит:
  - Time-series only baseline: **+10% AUC**
  - Time-series + static patient data: **+5% AUC**
  - SOTA post-KTx model: graft loss 0.96 vs 0.94, rejection 0.84 vs 0.74
- **Модальности:** Irregular time series + unstructured text

> 🟢 **Ключевой количественный результат:** добавление текстовой модальности к временным рядам + статике даёт +5-10% AUC. Это прямой аргумент за мультимодальный подход.

---

## 4. 🧩 Обработка пропущенных модальностей

### TRACE: Temporal Conditional Estimation for Multimodal TS Foundation Models
- 📅 **2026-06-04** | 🔗 `arxiv.org/abs/2606.06285`
- **Авторы:** Kan Z., Chen Y., Li K. et al. (Mayo Clinic)
- **Проблема:** В реальных мультимодальных данных временные ряды:
  - Имеют **temporal misalignment** (разные модальности — на разных временных шкалах)
  - Имеют **partial modality missingness** (часть модальностей отсутствует)
- **Решение:** TRACE — conditional estimation paradigm вместо наивной импутации
- **Инсайт:** «Existing approaches rely on naive imputation or masking strategies, which fail to account for cross-modal dependencies»

> 🟢 **Важно:** если у тебя ЧСС измеряется каждые 15 минут, а SpO2 — каждый час, это temporal misalignment. TRACE — state-of-the-art решение этой проблемы.

---

## 5. 🫀 Кардиология: мультимодальные подходы

### RAIM: Recurrent Attentive and Intensive Model of Multimodal Patient Monitoring
- 📅 **2018-07-23** | 🔗 `arxiv.org/abs/1807.08820`
- **Авторы:** Xu Y., Biswal S., Deshpande S.R., Maher K.O., Sun J. (Georgia Tech)
- **Модальности:** ECG, real-time vital signs, medications, discrete clinical events
- **Задача:** Детекция клинических событий в ICU
- **Архитектура:** RNN + attention для мультимодальных monitoring data
- **Инсайт:** «Integration of high-density monitoring data with discrete clinical events is challenging but potentially rewarding»

> 🟡 Классическая работа (2018), но архитектурный паттерн RNN+attention для мультимодальных ICU данных всё ещё актуален.

### ECG → Lab Abnormality Prediction (Multimodal: ECG + Metadata)
- 📅 **2024-11-22** | 🔗 `arxiv.org/abs/2411.14886`
- **Авторы:** Lopez Alcaraz J.M., Strodthoff N. (та же группа, что MDS-ICU)
- **Задача:** Предсказание 24+ лабораторных отклонений по ECG + demographics + vitals
- **Архитектура:** Structured State Space (S4) для ECG + late fusion для метаданных
- **Результат:** AUROC > 0.70 для большинства lab values
- **Охват:** Cardiac, renal, hematological, metabolic, immunological panels

---

## 6. 🧠 Representation Learning: общие подходы

### Multimodal Synthesis: MRI + Tabular via Cross-Attention in Joint Latent Space
- 📅 **2026-05-05** | 🔗 `arxiv.org/abs/2605.06699`
- **Авторы:** Mensing D., Kapar J., Hirsch J.G. et al.
- **Метод:** Diffusion model в shared latent space через cross-attention между MRI и табличными данными
- **Инсайт:** Cross-attention для табличных данных и изображений воспроизводим для таблиц + временных рядов

---

## 📊 Архитектурные паттерны: сводка

```
СТРАТЕГИИ СЛИЯНИЯ (FUSION):

1. EARLY FUSION (до экстрактора признаков)
   ┌──────┐  ┌──────┐
   │Static│  │  TS  │
   └──┬───┘  └──┬───┘
      └────┬────┘
      ┌────┴────┐
      │  Model  │
      └─────────┘
   ➤ Просто, но требует синхронизации размерностей

2. LATE FUSION (после экстракторов)
   ┌──────┐  ┌──────┐
   │Static│  │  TS  │
   │Encdr │  │Encodr│
   └──┬───┘  └──┬───┘
      └────┬────┘
      ┌────┴────┐
      │Fusion MLP│
      └─────────┘
   ➤ Лучше для гетерогенных модальностей (Shukla & Marlin 2020)

3. JOINT + LATE (MedPatch 2025)
   ┌──────┐  ┌──────┐
   │Static│  │  TS  │
   └──┬───┘  └──┬───┘
      ├── Joint Fusion ──┐
      └── Late Fusion ───┤
                    ┌────┴────┐
                    │Confidence│
                    │  Patching│
                    └─────────┘
   ➤ State-of-the-art: устойчиво к пропуску модальностей

4. CROSS-ATTENTION (TRACE 2026, TFN 2026)
   ┌──────┐       ┌──────┐
   │Static│ ←Q·K→ │  TS  │
   │Embds │       │Embds │
   └──────┘       └──────┘
   ➤ Модальности «общаются» друг с другом через attention
```

---

## 🎯 Практические рекомендации для твоего проекта

### Сценарий А: «Быстрый win» — статические агрегаты временных рядов
```
Сейчас:     [Killip, ХОБЛ, SpO2(t=0), ЧСС(t=0)] → CatBoost
Добавить:   [Killip, ХОБЛ, SpO2_mean, SpO2_trend, ЧСС_mean, ЧСС_trend] → CatBoost
```
Самый простой способ «добавить» временную компоненту — статистики временного ряда как новые табличные признаки. Не требует менять архитектуру.

### Сценарий B: «Late Fusion» — отдельные энкодеры
```
[Killip, ХОБЛ, TIMI] ─→ MLP/GB encoder ─┐
                                         ├─→ Fusion head → Prediction
[ЧСС(t), SpO2(t)] ───→ LSTM/GRU encoder ─┘
```
Late fusion (Shukla & Marlin 2020, Sadanandan 2026): обучаешь отдельные энкодеры для статики и динамики, сливаешь на последнем слое.

### Сценарий C: «MedPatch» — production-grade с учётом пропусков
```
[Killip, ХОБЛ] ─→ Static encoder ─┬── Joint fusion ─┐
[ЧСС(t), SpO2(t)] → TS encoder ───┴── Late fusion ──┤
                                                     ├─→ Confidence → Prediction
[Missing modality detector] ─────────────────────────┘
```
Если в real-world сценарии у части пациентов нет временных рядов (например, короткое пребывание) — MedPatch корректно обрабатывает пропуски модальностей.

---

## 📚 Cross-ссылки на другие файлы

| Тема | Где подробно |
|------|-------------|
| STAR-Set (асинхронные EHR) | `04_time_series_models.md` |
| Latent ODE (непрерывные ряды) | `04_time_series_models.md` |
| CTLPE (позиционное кодирование) | `04_time_series_models.md` |
| MDS-ICU (ECG + таблицы) | `04_time_series_models.md` и `07_fresh_papers_2024_2026.md` |
| Tabular Foundation Models | `07_fresh_papers_2024_2026.md` |
| Conformal Prediction (uncertainty) | `06_uncertainty_survival.md` |
| Missingness Indicators | `05_imputation_augmentation.md` |
