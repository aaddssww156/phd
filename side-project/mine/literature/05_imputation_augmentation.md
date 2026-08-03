# 🩻 Импутация и аугментация клинических данных

---

## 1. 🎯 Beyond Random Missingness: Clinically Rethinking Healthcare Time Series Imputation

**Библиография:**
> Qian L., Yang Y., Du W., Wang J., Dobson R., Ibrahim Z. (2024). Beyond Random Missingness: Clinically Rethinking for Healthcare Time Series Imputation. *arXiv:2405.17508*.

| Параметр | Значение |
|----------|----------|
| Проблема | Оценка импутации через random masking **не отражает** клиническую реальность |
| Данные | PhysioNet Challenge 2012 |
| Методы | 11 методов импутации |
| Вывод 1 | Способ маскирования **значимо влияет** на результаты |
| Вывод 2 | Точность импутации ≠ качество downstream prediction |
| Вывод 3 | RNN-архитектуры более робастны к разным паттернам пропусков |
| Ссылка | https://arxiv.org/abs/2405.17508 |

### Почему это важно для твоего проекта

Ты используешь MICE (через `miceforest`), который оценивается через cross-validation на импутированных данных. Но Qian et al. (2024) показывают:

> *«Imputation accuracy doesn't necessarily translate to optimal clinical prediction capabilities.»*

Клинические пропуски **не случайны** (MNAR, а не MAR/MCAR):
- SpO2 не измеряют, когда пациент стабилен → пропуск = информативный сигнал
- Лабораторные тесты назначают при подозрении на ухудшение → пропуск коррелирует с риском
- Катетеризацию не проводят при нестабильной гемодинамике → пропуск = тяжесть состояния

**Рекомендация:** добавить **missingness indicators** (бинарные признаки «было ли измерение») как отдельные предикторы — они несут клиническую информацию.

---

## 2. 🔁 Time-Dependent Iterative Imputation (TDI)

**Библиография:**
> Noy O., Shamir R. (2023). Time-dependent Iterative Imputation for Multivariate Longitudinal Clinical Data. *arXiv:2304.07821*.

| Параметр | Значение |
|----------|----------|
| Задача | Импутация многомерных лонгитюдных клинических данных |
| Идея | Forward-filling + Iterative Imputer с динамическими весами |
| Веса | Учитывают: missing rate, частоту измерений, паттерны |
| Данные | MIMIC-III: >500 000 наблюдений |
| Результат | RMSE 0.63 vs 0.85 (SoftImpute) на 25/30 переменных |
| Ссылка | https://arxiv.org/abs/2304.07821 |

### TDI vs MICE (твой подход)

| | MICE (miceforest) | TDI |
|---|---|---|
| **Тип данных** | Статические (одна строка = пациент) | Лонгитюдные (много строк = пациент) |
| **Учёт времени** | Нет | Динамические веса по времени |
| **Метод** | Chained Equations + LightGBM | Forward-filling + Iterative Imputer |
| **Когда использовать** | Один срез на пациента | Временные ряды |

Если твои данные — один статический срез (как в `baseline.ipynb`), MICE — правильный выбор. Если добавишь временную компоненту, TDI может быть лучше.

---

## 3. 🏥 Missingness as Stability: Пропуски как информация

**Библиография:**
> Fleming S.L., Jeyapragasan K., Duan T. et al. (2019). Missingness as Stability: Understanding the Structure of Missingness in Longitudinal EHR Data. *arXiv:1911.07084*.

| Параметр | Значение |
|----------|----------|
| Проблема | LOCF (last observation carried forward) уничтожает информацию о missingness |
| Идея | Сохранять missingness pattern как часть представления пациента |
| Результат | Альтернативное представление consistently better для optimal control |
| Ссылка | https://arxiv.org/abs/1911.07084 |

### Ключевой инсайт
В клинических данных **отсутствие измерения — это сигнал**. Если пациенту не измеряют SpO2 каждые 15 минут, потому что он стабилен — это позитивный сигнал. Если не измеряют lactate — это может быть негативным (нет времени/возможности). LOCF и простая импутация median/mean этот сигнал уничтожают.

### Практическая рекомендация
Для каждого признака с пропусками добавить бинарный индикатор `{feature}_missing`:
- `ЧСС_missing = 1` → измерение не проводилось → возможно, пациент стабилен
- `SpO2_missing = 1` → не измеряли → но это может быть плохим знаком в реанимации

Это даёт модели дополнительную информацию о процессе сбора данных.

---

## 4. 📊 Tabular Data Augmentation (обзор)

**Библиография:**
> Cui L., Li H., Chen K., Shou L., Chen G. (2024). Tabular Data Augmentation for Machine Learning: Progress and Prospects of Embracing Generative AI. *arXiv:2407.21523*.

| Параметр | Значение |
|----------|----------|
| Тип | Comprehensive survey |
| Охват | Retrieval-based + Generation-based методы |
| Гранулярность | Row, column, cell, table уровни |
| Ресурс | github.com/SuDIS-ZJU/awesome-tabular-data-augmentation |
| Ссылка | https://arxiv.org/abs/2407.21523 |

### Методы аугментации для клинических табличных данных

| Метод | Тип | Плюсы | Минусы |
|-------|-----|-------|--------|
| SMOTE | Generation (row) | Простой, популярный | Не работает с категориальными, создаёт нереалистичные сэмплы |
| CTGAN | Generation (table) | SOTA GAN для таблиц | Сложен в настройке |
| **VAE-GMM** | Generation (table) | Лучше CTGAN для медицины | Новый, меньше adoption |
| Bootstrap | Retrieval (row) | Простой, сохраняет распределение | Не добавляет новизны |

---

## 5. 🧬 VAE-GMM: улучшенный генератор табличных данных

**Библиография:**
> Apellániz P.A., Parras J., Zazo S. (2024). An Improved Tabular Data Generator with VAE-GMM Integration. *arXiv:2404.08434*.

| Параметр | Значение |
|----------|----------|
| Идея | Bayesian Gaussian Mixture (BGM) внутри VAE вместо одного Gaussian |
| Проблема CTGAN/TVAE | Предполагают гауссово latent space → плохо для не-гауссовых клинических данных |
| Результат | Значимое превосходство над CTGAN и TVAE на медицинских датасетах |
| Ссылка | https://arxiv.org/abs/2404.08434 |

### Зачем это может понадобиться
Если у тебя сильный дисбаланс классов (КШ — редкое событие), VAE-GMM может генерировать синтетические случаи КШ для:
1. Аугментации обучающей выборки
2. Стресс-тестирования модели на редких комбинациях признаков

Но с учётом выводов Sirikul et al. (2026) о вреде коррекции дисбаланса — использовать осторожно и только с калибровочной проверкой.

---

## 6. 🤖 Bayesian Recurrent Framework: импутация + предсказание одновременно

**Библиография:**
> Guo Y., Liu Z., Krishnswamy P., Ramasamy S. (2019). Bayesian Recurrent Framework for Missing Data Imputation and Prediction with Clinical Time Series. *arXiv:1911.07572*.

| Параметр | Значение |
|----------|----------|
| Идея | Единая байесовская RNN для имитации + предсказания |
| Преимущество | Учитывает неопределённость (не детерминированные импутации) |
| Данные | MIMIC-III, PhysioNet |
| Ссылка | https://arxiv.org/abs/1911.07572 |

### Отличие от MICE
MICE (твой подход) даёт **детерминированные** импутации (после усреднения по 5 итерациям). Bayesian framework даёт **распределение** возможных значений — это позволяет оценить, насколько импутация неопределённа, и как это влияет на финальное предсказание.

---

## 7. 🔬 MICE-RF vs Deep Learning для медицинских временных рядов

**Библиография:**
> Le L.P., Nguyen Thi X.-H., Nguyen T. et al. (2024). Missing Data Imputation for Noisy Time-Series Data and Applications in Healthcare. *arXiv:2412.11164*.

| Параметр | Значение |
|----------|----------|
| Сравнение | MICE-RF vs SAITS, BRITS, Transformer |
| Метрики | MAE, F1, AUC, MCC |
| Диапазон | 10%–80% пропусков |
| Результат | MICE-RF эффективно импутирует и имеет **denoising effect** |
| Ссылка | https://arxiv.org/abs/2412.11164 |

### Важный вывод для тебя
> *«MICE-RF can effectively impute missing data compared to deep learning methods and the improvement in classification of data imputed indicates that imputation can have denoising effects.»*

Твой выбор MICE (через miceforest/LightGBM) подтверждается: на практике MICE часто превосходит сложные DL-методы для импутации клинических данных, особенно при умеренном проценте пропусков. Плюс — denoising эффект: импутация сглаживает шум.

---

## 📊 Сводка: что делать с пропусками

| Стратегия | Когда применять | Что даёт |
|-----------|-----------------|----------|
| **Missingness indicators** | Всегда | Сигнал о процессе сбора данных |
| MICE (miceforest) | Статические данные (твой случай) | Лучшее качество импутации |
| TDI (Noy 2023) | Лонгитюдные данные | Учёт временной структуры |
| Forward-filling + indicator | Быстрый baseline | Сравнение с MICE |
| Bayesian imputation | Нужна uncertainty | Распределение, а не точка |
