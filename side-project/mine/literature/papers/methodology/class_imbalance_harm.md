---
title: "Class Imbalance Corrections Failed to Enhance Discrimination, Model Calibration, and Prediction Stability"
authors: "Sirikul W., Isaradech N., Kiratipaisarl W., Wongyikul P., Jirattikanwong N., Phinyo P."
year: 2026
published: "2026-06-08"
arxiv_id: "2606.08966"
tags: [class-imbalance, calibration, clinical-prediction, bootstrap, simulation]
category: "methodology"
---

# Class Imbalance Corrections Failed to Enhance Discrimination, Model Calibration, and Prediction Stability

**Авторы:** Sirikul W., Isaradech N., Kiratipaisarl W., Wongyikul P., Jirattikanwong N., Phinyo P.

**Дата:** 2026-06-08 | **arXiv:** [2606.08966](https://arxiv.org/abs/2606.08966) | **PDF:** [Скачать](https://arxiv.org/pdf/2606.08966)

---

## Аннотация

Class imbalance is common in clinical prediction models. This study investigated how imbalance correction affects classification performance and prediction stability using GUSTO-I (40,830 patients, 2,851 events). All imbalance-correction strategies led to miscalibration, risk overestimation, and increased prediction instability. Class imbalance should not be treated as a pathology that automatically requires correction.

---

## Релевантность проекту

КРИТИЧЕСКИ ВАЖНО: твой class_weight='balanced', scale_pos_weight, auto_class_weights могут вредить калибровке. Рекомендуется ablation study без коррекции.

---

## Методологические заметки

Penalised logistic regression, algorithm-level + data-level correction, 200 bootstrap resamples, MAPE, CII

---

## Связанные статьи

[[2408.01626]], [[2504.04906]], [[2504.16185]]

---

## Ключевые теги

#class-imbalance #calibration #clinical-prediction #bootstrap #simulation
