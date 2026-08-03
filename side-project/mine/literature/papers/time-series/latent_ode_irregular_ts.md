---
title: "Latent ODEs for Irregularly-Sampled Time Series"
authors: "Rubanova Y., Chen R.T.Q., Duvenaud D."
year: 2019
published: "2019-07-08"
arxiv_id: "1907.03907"
tags: [neural-ODE, irregular-time-series, RNN, continuous-time, latent-variables]
category: "time-series"
---

# Latent ODEs for Irregularly-Sampled Time Series

**Авторы:** Rubanova Y., Chen R.T.Q., Duvenaud D.

**Дата:** 2019-07-08 | **arXiv:** [1907.03907](https://arxiv.org/abs/1907.03907) | **PDF:** [Скачать](https://arxiv.org/pdf/1907.03907)

---

## Аннотация

Time series with non-uniform intervals occur in many applications, and are difficult to model using standard RNNs. We generalize RNNs to have continuous-time hidden dynamics defined by ODEs (ODE-RNNs). Furthermore, we use ODE-RNNs to replace the recognition network of Latent ODE. Both can naturally handle arbitrary time gaps and explicitly model the probability of observation times using Poisson processes.

---

## Релевантность проекту

Фундаментальная работа. Непрерывная динамика через ОДУ позволяет обрабатывать произвольные промежутки между измерениями — именно то, что нужно для данных ОРИТ (ЧСС, SpO2 снимаются с разной частотой).

---

## Методологические заметки

ODE-RNN, Latent ODE, Poisson process likelihood, VAE framework

---

## Связанные статьи

[[2409.20092]], [[2603.06605]], [[2410.01086]]

---

## Ключевые теги

#neural-ODE #irregular-time-series #RNN #continuous-time #latent-variables
