---
tags: [index, literature, cardiogenic-shock]
---

# 📚 Индекс статей

> Всего: **72 статьи** в 10 категориях.
> Для каждой статьи — отдельный файл с аннотацией, релевантностью проекту и ссылками на связанные работы.

---

## 🫀 Кардиогенный шок

- [[cshock_dynamic_risk_score]] — CShock: DL risk score, AUROC 0.82 (2023)
- [[guardian_rl_cardiogenic_shock]] — Safe RL для weaning MCS при КШ (2025)

## 🏥 ML в ICU

- [[xgboost_icu_hf_mimic]] — XGBoost + SHAP, MIMIC-III, AUROC 0.92 (2024)
- [[xmi_icu_heart_attack]] — Time-resolved SHAP, eICU+MIMIC-IV (2023)
- [[catboost_elderly_icu_dm_hf]] — CatBoost + DREAM, AUROC 0.86 (2025)
- [[early_mortality_htn_af_icu]] — MICE → ML, первые 24h ICU (2025)
- [[sepsis_mortality_mimic]] — Sepsis, та же методология Pishgar (2024)
- [[xgboost_mods_elderly_multicenter]] — Multicenter: 3 базы, XGBoost+SHAP (2020)
- [[multimodal_mortality_multicenter_4dbs]] — 4 базы, multimodal, external val. (2025)

## ⚙️ Методология

- [[class_imbalance_harm]] — 🔴 Коррекция дисбаланса вредит калибровке (2026)
- [[weighted_brier_score]] — Weighted Brier + clinical utility (2024)
- [[brier_misconceptions]] — Ошибки интерпретации Brier score (2025)
- [[mice_vs_deterministic_imputation]] — MICE vs детерминированная (2024)
- [[rare_events_metrics]] — AUC misleading при редких событиях (2025)

## ⏱️ Временные ряды

- [[latent_ode_irregular_ts]] — Neural ODE для нерегулярных рядов (2019)
- [[ctlpe_irregular_ts]] — Непрерывное позиционное кодирование (2024)
- [[star_set_async_ehr]] — STAR-Set: attention biases для EHR (2026)
- [[multimodal_icu_deterioration_bilstm]] — BiLSTM + ClinicalBERT, 5.7M сэмплов (2026)
- [[ts_clinical_notes_fusion]] — Early vs late fusion TS+текст (2020)
- [[trace_multimodal_ts_fm]] — TRACE: conditional estimation missing modalities (2026)

## 🔗 Мультимодальный fusion

- [[medpatch_multimodal_fusion]] — MedPatch: multi-stage + confidence (2025)
- [[mind_knowledge_distillation]] — MIND: KD для multimodal clinical (2025)
- [[static_mts_fusion_amr]] — Static + MTS fusion (2024)
- [[tfn_temporal_fusion_nexus]] — TFN: +10% AUC от multimodal (2026)
- [[mds_icu_multimodal_deterioration]] — MDS-ICU: 33 outcomes, S4+RealMLP (2026)
- [[ecg_lab_abnormality_prediction]] — ECG→Labs: S4 + late fusion (2024)
- [[raim_multimodal_monitoring]] — RAIM: RNN+attention multimodal ICU (2018)

## 🎯 Uncertainty & Survival

- [[conformal_prediction_intro]] — Conformal prediction: tutorial (2021)
- [[self_calibrating_conformal]] — Venn-Abers + conformal (2024)
- [[deep_survival_analysis_monograph]] — Deep survival analysis (2024)
- [[conformal_gb_nafld]] — GB + conformal prediction (2026)
- [[conformal_biomarker_trajectories]] — Conformal bands для биомаркеров (2025)
- [[siamese_survival_competing_risks]] — Competing risks (2018)
- [[deep_bayesian_gp_ehr]] — Deep Bayesian GP для EHR (2020)

## 🩻 Импутация и аугментация

- [[beyond_random_missingness]] — Random masking невалидна (2024)
- [[tdi_time_dependent_imputation]] — TDI: time-dependent imputation (2023)
- [[closing_gaps_icu_imputation]] — ICU vital signs imputation benchmark (2025)
- [[mice_rf_vs_dl_imputation]] — MICE-RF vs DL: denoising effect (2024)
- [[missingness_as_stability]] — Missingness = сигнал (2019)
- [[bayesian_recurrent_imputation]] — Bayesian imputation+prediction (2019)
- [[tabular_data_augmentation_survey]] — TDA survey (2024)
- [[vae_gmm_tabular_generator]] — VAE-GMM генератор таблиц (2024)

## 🆕 Свежие 2024-2026

- [[tabular_fm_survival]] — Tabular FM + survival (2026)
- [[retrieval_tabular_fm_ehr]] — TICL деградирует при дисбалансе (2026)
- [[tabular_llms_alzheimer]] — LLM для табличных клинических данных (2026)
- [[drum_transfer_cardiac_arrest]] — DRUM: transfer learning + missing covariates (2026)
- [[hcm_risk_score_ml]] — ML risk score vs ESC score (2026)
- [[t2d_external_validation_fairness]] — Multi-dim evaluation framework (2026)
- [[disentangling_multimodal_clinical]] — Multi-task disentangled clinical (2026)

## 🧩 Дополнительные углы

- [[shap_optuna_xgb_lgbm_catboost]] — 🔥 Идентичный техностек (2025)
- [[fairness_icu_mortality]] — Fairness monitoring ICU (2024)
- [[llm_fairness_icu]] — LLM debiasing ICU (2025)
- [[variational_disentanglement_rare_events]] — Альтернатива class_weight (2020)
- [[surgvae_cardiac_complications]] — surgVAE cardiac surgery (2024)
- [[healthcare_ai_mlops_framework]] — MLOps: 5 pillars (2025)
- [[agentic_ai_medicine_review]] — Agentic AI: scoping review (2026)

## 🧱 Расширенный охват

- [[flicu_federated_icu]] — FLICU: federated learning (2022)
- [[federated_oneflorida_postop]] — OneFlorida+ 358K пациентов (2026)
- [[automl_med_tabular]] — AutoML-Med (2025)
- [[climb_clinical_automl]] — CliMB (van der Schaar, 2024)
- [[causal_inference_medicine_summary]] — Causal inference in medicine (2021)
- [[causal_effects_ehr_dl]] — Causal effects + DL on EHR (2020)
- [[rashomon_effect_clinical]] — Rashomon Effect: выбор из 5 моделей (2025)
- [[domain_shifts_clinical_models]] — Domain shifts clinical (2018)
- [[domain_invariant_ehr_representations]] — Domain-invariant representations (2023)
- [[inadequacy_stochastic_nn_clinical]] — Stochastic NN неадекватны (2024)
- [[early_prediction_icu_review]] — First-day ICU review (2025)
- [[orthkd_clinical_kd]] — KD для clinical deployment (2026)
- [[qi_smote_imbalanced_medical]] — QI-SMOTE (2025)
- [[cvd_detection_ml_comparison]] — ML comparison CVD (2024)
- [[mri_tabular_diffusion_cross_attention]] — Cross-attention MRI+tabular (2026)
- [[ehr_ragp_foundation_model]] — EHR-RAGp: RAG foundation model (2026)
