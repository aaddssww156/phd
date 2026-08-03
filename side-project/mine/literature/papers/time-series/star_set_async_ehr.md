---
title: "Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series"
authors: "Lee J., Lee K., Kim C., Yang E."
year: 2026
published: "2026-02-18"
arxiv_id: "2603.06605"
tags: [EHR, set-transformer, attention, asynchronous-time-series, ICU, foundation-model]
category: "time-series"
---

# Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series

**Авторы:** Lee J., Lee K., Kim C., Yang E.

**Дата:** 2026-02-18 | **arXiv:** [2603.06605](https://arxiv.org/abs/2603.06605) | **PDF:** [Скачать](https://arxiv.org/pdf/2603.06605)

---

## Аннотация

EHR are irregular, asynchronous multivariate time series. STAR-Set adds parameter-efficient soft attention biases: a temporal locality penalty with learnable timescales and a variable-type affinity from a learned feature-compatibility matrix. On three ICU prediction tasks, STAR-Set achieves AUC/APR of 0.716/0.003 (CPR), 0.916/0.203 (mortality), 0.837/0.126 (vasopressor).

---

## Релевантность проекту

State-of-the-art для асинхронных EHR. Attention biases решают проблему grid vs point-set токенизации. AUC 0.916 на mortality — сильный baseline.

---

## Методологические заметки

Temporal locality bias (−|Δt|/τ), variable-type affinity matrix, 10 fusion schedules benchmarked

---

## Связанные статьи

[[1907.03907]], [[2409.20092]], [[2606.12006]]

---

## Ключевые теги

#EHR #set-transformer #attention #asynchronous-time-series #ICU #foundation-model
