---
title: "A Multimodal Deep Learning Framework for Predicting ICU Deterioration: Integrating ECG Waveforms with Clinical Data"
authors: "Lopez Alcaraz J.M., Lopez Moran X., Davila Zaragoza E., Handel C., Koebe R., Haverkamp W., Strodthoff N."
year: 2026
published: "2026-01-10"
arxiv_id: "2601.06645"
tags: [ICU, multimodal, ECG, S4-state-space, RealMLP, MIMIC-IV]
category: "multimodal-fusion"
---

# A Multimodal Deep Learning Framework for Predicting ICU Deterioration: Integrating ECG Waveforms with Clinical Data

**Авторы:** Lopez Alcaraz J.M., Lopez Moran X., Davila Zaragoza E., Handel C., Koebe R., Haverkamp W., Strodthoff N.

**Дата:** 2026-01-10 | **arXiv:** [2601.06645](https://arxiv.org/abs/2601.06645) | **PDF:** [Скачать](https://arxiv.org/pdf/2601.06645)

---

## Аннотация

MDS-ICU fuses demographics, biometrics, vital signs, labs, ECG waveforms, surgical procedures, and medical device usage to predict 33 clinically relevant outcomes. S4 encoders for ECG + RealMLP for tabular data. AUROCs: 0.90 (24h mortality), 0.92 (sedation), 0.97 (ventilation), 0.93 (coagulation). Model outperformed clinicians and LLMs.

---

## Релевантность проекту

Multimodal fusion работает: даже без ECG, подход «структурированные данные + временные ряды» даёт AUROC 0.90. 33 outcomes — multi-task.

---

## Методологические заметки

S4 state-space encoder, RealMLP, multi-task (33 outcomes), clinician benchmarking

---

## Связанные статьи

[[2603.14719]], [[2411.14886]], [[2003.11059]]

---

## Ключевые теги

#ICU #multimodal #ECG #S4-state-space #RealMLP #MIMIC-IV
