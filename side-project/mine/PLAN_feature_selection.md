# Feature Selection Pipeline — Полный план

> Создан: 2026-08-03
> Проект: prediction cardiogenic shock (КШ) в реанимации, табличные данные
> Целевая переменная: "КШ развился в реанимации"

---

## 1. Загрузка и подготовка данных

### 1.1. Фильтры DataSet_V49
```python
data = dataAllFull.loc[
    (dataAllFull['STEMI (Новый)'] == 'Да') &
    (dataAllFull['Наличие в файле'] == 'Да') &
    (dataAllFull['ЧКВ'] == 'Да')
].copy()
```

### 1.2. Мerging с filtered_killip
```python
key = 'Код пациента'
final_df = data.merge(killip_sub, on=key, how='inner')
```

### 1.3. MICE импутация
```python
mice_cols = [...]  # user specifies
kernel = mf.ImputationKernel(data=final_df[mice_cols], random_state=42)
kernel.mice(iterations=10)
final_df[mice_cols] = kernel.complete_data(dataset=0).values
```
**Global fit** — MICE применяется до сплита. Пользователь согласовал с оговоркой о потенциальной оптимизме метрик.

### 1.4. Разделение признаков
```python
X_orig = final_df[FEATURES_TO_SELECT].copy()  # DataFrame
y = final_df[TARGET].astype(int)
groups = final_df[key]  # для GroupKFold
```

---

## 2. Предобработка (fit один раз на full data)

### 2.1. Для LogisticRegression
```python
lr_pipe = Pipeline([
    ('pre', ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ]))
])
X_lr = lr_pipe.fit_transform(X_orig)
```

### 2.2. Для XGBoost
```python
xgb_pipe = Pipeline([
    ('pre', ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ]))
])
X_xgb = xgb_pipe.fit_transform(X_orig)
```

### 2.3. Feature names для LR (SHAP/importance)
```python
lr_all_names = list(lr_pipe.named_steps['pre'].get_feature_names_out())
```
**КРИТИЧНО:** `.get_feature_names_out()` — не `.named_transformers_['cat']`.

**Важно:** После этого все selection методы работают на **numpy arrays** (`X_lr`, `X_xgb`), а не на DataFrame.

---

## 3. CV Setup
```python
gkf = GroupKFold(n_splits=5)
scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': make_scorer(average_precision_score),
    'brier': make_scorer(brier_score_loss, needs_threshold=True),
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
}
```
- `class_weight='balanced'` для всех моделей
- groups = всегда передается, никогда не режется

---

## 4. Baseline — Optuna tuning

### 4.1. LR
```python
params = {'C': log-uniform(1e-3, 1e2), 'penalty': 'l2', 'solver': 'lbfgs',
          'max_iter': 1000, 'class_weight': 'balanced'}
```

### 4.2. XGB
```python
params = {'n_estimators': 100-500, 'max_depth': 3-8,
          'learning_rate': log-uniform(1e-3, 0.3),
          'colsample_bytree': 0.5-1.0,
          'reg_alpha': 0-1.0, 'reg_lambda': 0-1.0,
          'use_label_encoder': False, 'eval_metric': 'logloss',
          'random_state': 42, 'class_weight': 'balanced'}
```

### 4.3. Tuning loop
```python
cross_validate(clf, X_lr/X_xgb, y, groups=groups, cv=gkf, scoring='roc_auc')
```

---

## 5. RFE — Rank by Feature Importance

### 5.1. Реализация
```python
pipe = Pipeline([('pre', lr_pipe/xgb_pipe), ('clf', base_clf)])
rfe = RFE(pipe, step=1, n_features_to_select=1)
rfe.fit(X_data, y)  # X_data — numpy array
ranking_ → original features
```

### 5.2. Pipeline wrapper
- `Pipeline([('pre', preprocessor), ('clf', estimator)])`
- RFE.fit проходит через preprocessor → `ranking_` ссылается на исходные колонки
- Для итераций k: `X_data[:, ranked[n-k:]]` — срез numpy array

### 5.3. Оценка
```python
cross_validate(clf_slim, X_sliced, y, groups=groups, cv=gkf, scoring=scoring)
```
groups передается полный — никогда не режется.

---

## 6. Greedy Forward / Backward

### 6.1. Forward (start empty)
```python
selected = []
for step in range(1, n+1):
    for f in range(n):
        if f in selected: continue
        sub = selected + [f]
        score = mean(cv_scores(model, X[:, sub], groups=groups))
    add best f to selected
```

### 6.2. Backward (start full)
```python
selected = list(range(n))
for step in range(1, n+1):
    for f in range(n):
        if f not in selected: continue
        sub = [x for x in selected if x != f]
        if not sub: continue
        score = mean(cv_scores(model, X[:, sub], groups=groups))
    remove worst f from selected
    if not selected: break
```

### 6.3. Ключевое
- Работает на **numpy arrays** `X_lr[:, sub]`, `X_xgb[:, sub]`
- groups передается полный, never sliced
- Каждая итерация — fresh estimator, never reused

---

## 7. Результаты

### 7.1. CSV output
```
results/feature_selection_<timestamp>.csv
```
Колонки: Method, Step, N_Features, Features, ROC_AUC, PR_AUC, Brier, Precision, Recall, F1

### 7.2. Best model → importance
```python
bmodel = ob['Method'].split('_')[-1]
bfeat_idx = eval(ob['Features'])
Xb = X_orig.iloc[:, bfeat_idx]

# Refit preprocessing on best subset
Xb_pre = best_pipe.fit_transform(Xb)
base_clf.fit(Xb_pre, y)

# SHAP для XGBoost
expl = shap.TreeExplainer(base_clf)
sv = expl.shap_values(Xb_pre)

# Coefficients для LR
fi = pd.DataFrame({'Feature': fnames, 'Coefficient': base_clf.coef_[0]})
```

---

## 8. Известные проблемы и решения

| Проблема | Решение |
|----------|---------|
| `NotFittedError` на `get_feature_names_out()` | Использовать `.named_steps['pre'].get_feature_names_out()` (на ColumnTransformer), НЕ `.named_transformers_['cat']` |
| `make_column_selector` ломается на subset | Не использовать — fit один раз на full, потом numpy array |
| ColumnTransformer падает на подмножестве колонок | Не нужен — всё работает на numpy |
| RFE `ranking_` ссылается на OHE-колонки | Pipeline wrapper: preprocessor + estimator, ranking_ на оригинальных |
| `list.remove(x): x not in list` в Backward | Проверять `if not sub: continue` в inner loop |
| Cell 0 перезаписывает config | Cell 0 содержит только импорты, config задаётся пользователем |

---

## 9. Статус

- ✅ Data loading & merge
- ✅ MICE imputation (global)
- ✅ Preprocessing (one-time fit, numpy arrays)
- ✅ Optuna baseline (LR + XGB)
- ✅ RFE (Pipeline wrapper, numpy slicing)
- ✅ Greedy Forward (numpy slicing)
- ⏳ Greedy Backward — test shows `list.remove(x): x not in list` — fix: check `if not sub: continue`
- ⏳ Save results + importance — pending

## 10. Архитектура файлов

```
feature-selection.ipynb
├── Cell 0: imports (NO variable assignments)
├── Cell 1: load, merge, MICE, preprocess → X_lr, X_xgb
├── Cell 2: Baseline (Optuna) + RFE
└── Cell 3: Greedy (fwd+bwd) + save + importance
```