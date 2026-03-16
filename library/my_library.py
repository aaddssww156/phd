import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.metrics import (roc_auc_score, f1_score, precision_recall_curve, auc, confusion_matrix, recall_score, precision_score, balanced_accuracy_score)
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint

def do_magic(df): 
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 1000)

    display(Markdown("## 📊 Базовая информация о датасете"))

    display(Markdown("### 💾 Информация о типах данных"))
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    display(Markdown(f"```\n{info_str}\n```"))

    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'First Value': df.iloc[0].values,
        'Last Value': df.iloc[-1].values,
        'Unique Values': df.nunique().values
    })

    display(info_df.style
            .set_caption("DataFrame Schema Overview")
            .format(precision=2)
            .background_gradient(cmap='Blues', subset=['Unique Values']))

    display(Markdown("### 📈 Описательная статистика числовых признаков"))
    display(df.describe().T.style.background_gradient(cmap='viridis'))

    display(Markdown("### 👀 Первые 5 строк данных"))
    display(df.head())

    display(Markdown("### 👀 Последние 5 строк данных"))
    display(df.tail())

    display(Markdown("### 🔍 Анализ уникальных значений (первые 10 столбцов)"))
    unique_counts = {}
    for col in df.columns[:10]:
        unique_counts[col] = {
            'unique_count': df[col].nunique(),
            'unique_values': df[col].unique()[:10].tolist(),
            'dtype': str(df[col].dtype)
        }

    unique_df = pd.DataFrame(unique_counts).T
    display(unique_df)

    target_variable = 'Смерть'
    df[target_variable] = df['Смерть'].map({'Да': 1, 'Нет': 0})

    class_counts = df[target_variable].value_counts()
    class_percentages = (class_counts / len(df)) * 100

    display(Markdown("## ⚖️ Анализ дисбаланса классов"))
    display(Markdown(f"### Целевая переменная: `{target_variable}`"))

    balance_df = pd.DataFrame({
        'Класс': class_counts.index,
        'Количество': class_counts.values,
        'Процент': class_percentages.values
    })
    display(balance_df.style.background_gradient(cmap='Reds'))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_counts.index.astype(str), class_counts.values, 
                color=['#ff9999', '#66b3ff'], alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:,}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12)

    ax.set_title('Дисбаланс классов', fontsize=15, fontweight='bold')
    ax.set_xlabel('Класс', fontsize=12)
    ax.set_ylabel('Количество наблюдений', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    imbalance_ratio = max(class_counts.values) / min(class_counts.values)
    display(Markdown(f"### 📈 Степень дисбаланса: {imbalance_ratio:.1f}:1"))
    if imbalance_ratio > 5:
        display(Markdown(f"⚠️ **Сильный дисбаланс!** Требуются специальные методы обработки."))
    elif imbalance_ratio > 2:
        display(Markdown(f"🟡 **Умеренный дисбаланс.** Нужно учитывать при выборе метрик."))

    display(Markdown("## 🕳️ Анализ пропущенных значений"))

    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Столбец': missing_percentages.index,
        'Процент пропусков': missing_percentages.values
    }).sort_values('Процент пропусков', ascending=False)

    missing_df = missing_df[missing_df['Процент пропусков'] > 0]

    display(Markdown(f"### 📊 Общая статистика по пропускам:"))
    display(Markdown(f"- Всего столбцов с пропусками: {len(missing_df)} из {len(df.columns)}"))
    display(Markdown(f"- Средний процент пропусков: {missing_df['Процент пропусков'].mean():.2f}%"))
    display(Markdown(f"- Максимальный процент пропусков: {missing_df['Процент пропусков'].max():.2f}%"))

    display(Markdown("### 🏆 Топ-20 столбцов с наибольшими пропусками:"))
    display(missing_df.head(20).style.background_gradient(cmap='YlOrRd', subset=['Процент пропусков']))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(missing_df['Процент пропусков'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Распределение процентов пропусков по столбцам', fontsize=15, fontweight='bold')
    ax.set_xlabel('Процент пропусков', fontsize=12)
    ax.set_ylabel('Количество столбцов', fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    if target_variable in df.columns:
        display(Markdown("### 🎯 Анализ пропусков в контексте целевой переменной"))
        
        key_missing_cols = missing_df.head(5)['Столбец'].tolist()
        
        for col in key_missing_cols:
            if col != target_variable:
                display(Markdown(f"#### Столбец: `{col}`"))
                
                missing_analysis = pd.crosstab(
                    df[target_variable],
                    df[col].isnull().map({True: 'Пропуск', False: 'Значение'}),
                    margins=True,
                    margins_name='Всего'
                )
                
                missing_analysis_pct = missing_analysis.copy()
                for col_name in missing_analysis_pct.columns:
                    missing_analysis_pct[col_name] = (missing_analysis_pct[col_name] / 
                                                    missing_analysis_pct[col_name]['Всего'] * 100)
                
                display(Markdown("**Абсолютные значения:**"))
                display(missing_analysis)
                display(Markdown("**Проценты:**"))
                display(missing_analysis_pct.style.format("{:.1f}%"))

    numeric_cols = df.select_dtypes(include=['float64']).columns
    print(f"Собрано {len(numeric_cols)} цифровых колонок")

    display(Markdown("## 🔍 Поиск утечек данных (data leaks)"))

    display(Markdown("### 📈 Поиск признаков с крайне высокой корреляцией с target"))

    df_analysis = df.copy()

    cols_to_drop = df_analysis.columns[df_analysis.isnull().mean() == 1.0]
    if len(cols_to_drop) > 0:
        display(Markdown(f"Удаляем {len(cols_to_drop)} столбцов с 100% пропусков"))
        df_analysis = df_analysis.drop(columns=cols_to_drop)

    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()

    if target_variable in numeric_cols:
        numeric_cols.remove(target_variable)

    display(Markdown(f"### Числовые признаки для анализа: {len(numeric_cols)} из {len(df_analysis.columns)}"))

    if target_variable in df_analysis.columns and len(numeric_cols) > 0:
        if not pd.api.types.is_numeric_dtype(df_analysis[target_variable]):
            target_numeric = df_analysis[target_variable].astype('category').cat.codes
        else:
            target_numeric = df_analysis[target_variable]
        
        correlations = {}
        for col in numeric_cols:
            temp_df = df_analysis[[col, target_variable]].dropna()
            
            if len(temp_df) > 10:
                try:
                    corr = temp_df[col].corr(temp_df[target_variable])
                    if not np.isnan(corr):
                        correlations[col] = corr
                except:
                    continue
        if correlations:
            corr_df = pd.DataFrame({
                'Признак': list(correlations.keys()),
                'Корреляция с target': list(correlations.values())
            }).sort_values('Корреляция с target', key=abs, ascending=False)
            
            high_corr_threshold = 0.85
            potential_leaks = corr_df[abs(corr_df['Корреляция с target']) >= high_corr_threshold]
            
            display(Markdown("### 📊 Топ-20 признаков по корреляции с target:"))
            display(corr_df.head(20).style.background_gradient(cmap='coolwarm', subset=['Корреляция с target'], vmin=-1, vmax=1))
            
            if not potential_leaks.empty:
                display(Markdown(f"### ⚠️ **ПОТЕНЦИАЛЬНЫЕ УТЕЧКИ ДАННЫХ** (|корреляция| >= {high_corr_threshold}):"))
                display(potential_leaks.style.background_gradient(cmap='Reds', subset=['Корреляция с target']))
                
                plt.figure(figsize=(14, 8))
                top_corr = corr_df.head(25)
                bars = plt.barh(top_corr['Признак'], top_corr['Корреляция с target'], 
                            color=['#ff4444' if abs(x) >= high_corr_threshold else '#4488ff' for x in top_corr['Корреляция с target']])
                
                plt.axvline(x=high_corr_threshold, color='r', linestyle='--', alpha=0.7, label=f'Порог утечки ({high_corr_threshold})')
                plt.axvline(x=-high_corr_threshold, color='r', linestyle='--', alpha=0.7)
                
                plt.title(f'Топ-25 корреляций с целевой переменной "{target_variable}"', fontsize=15, fontweight='bold')
                plt.xlabel('Корреляция с target', fontsize=12)
                plt.ylabel('Признаки', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                display(Markdown("✅ Не обнаружено признаков с экстремально высокой корреляцией (потенциальных утечек)"))
        else:
            display(Markdown("❌ Не удалось вычислить корреляции. Проверьте типы данных и наличие пропусков."))


    display(Markdown("## 🏷️ Анализ категориальных признаков на предмет утечек"))

    categorical_cols = df_analysis.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_variable in categorical_cols:
        categorical_cols.remove(target_variable)

    display(Markdown(f"### Категориальные признаки для анализа: {len(categorical_cols)}"))

    if target_variable in df_analysis.columns and len(categorical_cols) > 0 and df_analysis[target_variable].nunique() == 2:
        display(Markdown("### 🔍 Поиск категориальных признаков с идеальным разделением классов"))
        
        potential_cat_leaks = []
        target_classes = df_analysis[target_variable].unique()
        target_classes = target_classes[~pd.isna(target_classes)]
        
        if len(target_classes) == 2:
            for col in categorical_cols:
                if df_analysis[col].nunique() < 20:
                    try:
                        cross_tab = pd.crosstab(df_analysis[col], df_analysis[target_variable])
                    except Exception as e:
                        print(col)
                        
                    if cross_tab.shape[0] > 1 and cross_tab.shape[1] == 2:
                        for category in cross_tab.index:
                            row = cross_tab.loc[category]
                            if (row.iloc[0] == 0 and row.iloc[1] > 0) or (row.iloc[0] > 0 and row.iloc[1] == 0):
                                if row.sum() > 10:
                                    leak_ratio = max(row) / row.sum()
                                    if leak_ratio > 0.95:
                                        potential_cat_leaks.append({
                                            'Признак': col,
                                            'Категория': str(category),
                                            'Распределение': row.values.tolist(),
                                            'Процент в одном классе': leak_ratio * 100
                                        })
        
        if potential_cat_leaks:
            leak_df = pd.DataFrame(potential_cat_leaks)
            display(Markdown(f"### ⚠️ **ПОТЕНЦИАЛЬНЫЕ УТЕЧКИ в категориальных признаках** (95%+ наблюдений в одном классе):"))
            display(leak_df.style.background_gradient(cmap='Reds', subset=['Процент в одном классе']))
        else:
            display(Markdown("✅ Не обнаружено категориальных признаков с идеальным разделением классов"))


    display(Markdown("## 📐 Анализ мультиколлинеарности признаков"))

    if len(numeric_cols) > 10:
        display(Markdown(f"⚠️ Слишком много числовых признаков ({len(numeric_cols)}) для полного анализа мультиколлинеарности"))
        display(Markdown("Будем анализировать только топ-50 признаков по корреляции с target"))
        
        top_features = corr_df.head(50)['Признак'].tolist() if 'corr_df' in locals() else numeric_cols[:50]
    else:
        top_features = numeric_cols

    if len(top_features) > 1:
        corr_matrix = df_analysis[top_features].corr().abs()
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=False,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": .5},
                    vmin=0,
                    vmax=1)
        
        plt.title('Матрица корреляций между признаками', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()
        
        high_corr_pairs = []
        threshold = 0.95
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'Признак 1': corr_matrix.columns[i],
                        'Признак 2': corr_matrix.columns[j],
                        'Корреляция': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Корреляция', ascending=False)
            display(Markdown(f"### ⚠️ Признаки с очень высокой корреляцией (> {threshold}):"))
            display(high_corr_df.style.background_gradient(cmap='Oranges', subset=['Корреляция']))
        else:
            display(Markdown("✅ Не обнаружено признаков с экстремально высокой корреляцией между собой"))


    display(Markdown("## 📈 Анализ распределений признаков по классам target"))

    if target_variable in df_analysis.columns and len(numeric_cols) > 0:
        top_analysis_features = corr_df.head(10)['Признак'].tolist() if 'corr_df' in locals() else numeric_cols[:10]
        
        if 'potential_leaks' in locals() and not potential_leaks.empty:
            leak_features = potential_leaks['Признак'].tolist()
            top_analysis_features = [f for f in top_analysis_features if f not in leak_features]
        
        if top_analysis_features:
            display(Markdown(f"### Распределения топ-{len(top_analysis_features)} признаков по классам target:"))
            
            n_features = len(top_analysis_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), 
                                    squeeze=False, tight_layout=True)
            
            target_classes = df_analysis[target_variable].unique()
            target_classes = target_classes[~pd.isna(target_classes)]
            
            for i, feature in enumerate(top_analysis_features):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                for class_val in target_classes:
                    class_data = df_analysis[df_analysis[target_variable] == class_val][feature]
                    class_data = class_data.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(class_data) > 10:
                        sns.kdeplot(class_data, ax=ax, label=f'Class {class_val}', 
                                alpha=0.7, linewidth=2)
                
                ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Значение', fontsize=10)
                ax.set_ylabel('Плотность', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
            for i in range(len(top_analysis_features), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.suptitle('Распределения признаков по классам target', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

    if 'df' not in globals():
        display(Markdown("❌ **Данные не загружены!** Выполните сначала ячейки с EDA и загрузкой данных."))
    else:
        display(Markdown("✅ **Данные доступны из предыдущих шагов**"))
        display(Markdown(f"Текущий размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов"))

    df_prep = df.copy()

    initial_count = len(df_prep)
    df_prep = df_prep.dropna(subset=[target_variable])
    removed_target_missing = initial_count - len(df_prep)

    if removed_target_missing > 0:
        display(Markdown(f"### 🗑️ Удалено {removed_target_missing} строк с пропусками в целевой переменной"))

    y = df_prep[target_variable].astype(int)
    class_counts = y.value_counts()
        
    display(Markdown("## ⚖️ Финальный дисбаланс классов:"))
    balance_df = pd.DataFrame({
        'Класс': class_counts.index,
        'Количество': class_counts.values,
        'Процент': (class_counts / len(y) * 100).values
    })
    display(balance_df.style.background_gradient(cmap='Reds'))

    if len(class_counts) < 2:
        display(Markdown("❌ **КРИТИЧЕСКАЯ ОШИБКА: Остался только один класс!**"))
    elif min(class_counts) < 10:
        display(Markdown(f"⚠️ **ОПАСНЫЙ ДИСБАЛАНС: В minority классе всего {min(class_counts)} наблюдений!**"))

    display(Markdown("## 🕳️ Шаг 2: Стратегическая обработка пропусков"))

    X = df_prep.drop(columns=[target_variable])

    missing_percentages = (X.isnull().sum() / len(X)) * 100
    high_missing_threshold = 70  # 70% пропусков как порог для удаления
    cols_to_drop_high_missing = missing_percentages[missing_percentages > high_missing_threshold].index.tolist()

    if cols_to_drop_high_missing:
        display(Markdown(f"### 🗑️ Удаляем {len(cols_to_drop_high_missing)} признаков с >{high_missing_threshold}% пропусков:"))
        display(Markdown(f"{cols_to_drop_high_missing[:10]}{'...' if len(cols_to_drop_high_missing) > 10 else ''}"))
        X = X.drop(columns=cols_to_drop_high_missing)
    else:
        display(Markdown("✅ Нет признаков для удаления по критерию >70% пропусков"))

    X_raw = X.copy()
    y_raw = y.copy()

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    display(Markdown(f"### 📊 Распределение типов признаков:"))
    display(Markdown(f"- **Числовых признаков:** {len(numeric_cols)}"))
    display(Markdown(f"- **Категориальных признаков:** {len(categorical_cols)}"))
    display(Markdown(f"- **Всего признаков:** {len(X_raw.columns)}"))

    if categorical_cols:
        display(Markdown("### 🔍 Примеры категориальных признаков:"))
        for col in categorical_cols[:5]:
            unique_vals = X_raw[col].dropna().unique()[:5]
            display(Markdown(f"- `{col}`: примеры значений {unique_vals}{'...' if len(unique_vals) > 5 else ''}"))
    else:
        display(Markdown("✅ Нет категориальных признаков для обработки"))

    display(Markdown("### 🔄 Кодирование категориальных признаков"))

    X_encoded = X_raw.copy()

    for col in categorical_cols:
        n_unique = X_encoded[col].nunique()
        n_samples = len(X_encoded)
        
        if n_unique == 2:
            unique_vals = X_encoded[col].dropna().unique()
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            X_encoded[col] = X_encoded[col].map(mapping)
            display(Markdown(f"✅ Бинарное кодирование: `{col}` → 0/1"))
            
        elif n_unique <= 10:
            dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[col])
            display(Markdown(f"✅ One-hot encoding: `{col}` → {len(dummies.columns)} новых признаков"))
            
        else:
            freq_map = X_encoded[col].value_counts(normalize=True)
            X_encoded[f"{col}_freq"] = X_encoded[col].map(freq_map)
            X_encoded = X_encoded.drop(columns=[col])
            display(Markdown(f"✅ Frequency encoding: `{col}` → частотное представление"))

    display(Markdown(f"### 📈 Размер после кодирования: {X_encoded.shape[0]} строк, {X_encoded.shape[1]} столбцов"))

    X = X_encoded.copy()

    empty_cols = X.columns[X.isnull().all()].tolist()
    if empty_cols:
        display(Markdown(f"Удаляем пустые признаки: {empty_cols}"))
        X = X.drop(columns=empty_cols)

    constant_cols = X.columns[X.nunique() == 1].tolist()
    if constant_cols:
        display(Markdown(f"Удаляем константные признаки: {constant_cols[:5]}{'...' if len(constant_cols) > 5 else ''}"))
        X = X.drop(columns=constant_cols)

    display(Markdown("### 🔍 Поиск признаков, присутствующих только в одном классе"))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    one_class_features = []

    for col in numeric_cols:
        class_0_data = X[y == 0][col]
        class_1_data = X[y == 1][col]
        
        count_0 = class_0_data.count()
        count_1 = class_1_data.count()
        
        if (count_0 == 0 and count_1 >= 5) or (count_1 == 0 and count_0 >= 5):
            one_class_features.append({
                'feature': col,
                'class_0_count': count_0,
                'class_1_count': count_1,
                'only_in_class': 1 if count_0 == 0 else 0
            })

    if one_class_features:
        one_class_df = pd.DataFrame(one_class_features)
        display(Markdown(f"### ⚠️ Найдено {len(one_class_features)} признаков, присутствующих только в одном классе:"))
        display(one_class_df.style.background_gradient(cmap='Reds'))
        
        features_to_drop = one_class_df['feature'].tolist()
        display(Markdown(f"**Удаляем {len(features_to_drop)} признаков как потенциальные скрытые утечки**"))
        X = X.drop(columns=features_to_drop)
    else:
        display(Markdown("✅ Не обнаружено признаков, присутствующих только в одном классе"))

    display(Markdown("## 🔍 Шаг 3: Отбор признаков"))

    display(Markdown("### 📐 Устранение мультиколлинеарности"))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    features_to_remove_corr = []

    if len(numeric_cols) > 5:
        X_corr = X[numeric_cols].copy()        
        X_corr_clean = X_corr
        
        if len(X_corr_clean) > 10:
            corr_matrix = X_corr_clean.corr().abs()
            
            high_corr_threshold = 0.85
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > high_corr_threshold:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
                display(Markdown(f"### Найдено {len(high_corr_pairs)} пар признаков с корреляцией > {high_corr_threshold}:"))
                
                for _, row in high_corr_df.iterrows():
                    if row['feature1'] in X.columns and row['feature2'] in X.columns:
                        temp_df1 = pd.concat([X[row['feature1']], y], axis=1).dropna()
                        temp_df2 = pd.concat([X[row['feature2']], y], axis=1).dropna()

                        if len(temp_df1) > 10 and len(temp_df2) > 10:
                            try:
                                feat1_num = pd.to_numeric(temp_df1.iloc[:, 0], errors='coerce')
                                feat2_num = pd.to_numeric(temp_df2.iloc[:, 0], errors='coerce')
                                target_num = pd.to_numeric(temp_df1.iloc[:, 1], errors='coerce')
                                
                                valid1 = pd.concat([feat1_num, target_num], axis=1).dropna()
                                valid2 = pd.concat([feat2_num, target_num], axis=1).dropna()
                                
                                if len(valid1) > 10 and len(valid2) > 10:
                                    corr1 = valid1.iloc[:, 0].corr(valid1.iloc[:, 1])
                                    corr2 = valid2.iloc[:, 0].corr(valid2.iloc[:, 1])
                                    
                                    feature_to_remove = row['feature1'] if abs(corr1) < abs(corr2) else row['feature2']
                                    if feature_to_remove not in features_to_remove_corr:
                                        features_to_remove_corr.append(feature_to_remove)
                            except:
                                continue
            
            if features_to_remove_corr:
                display(Markdown(f"### 🗑️ Удаляем {len(features_to_remove_corr)} мультиколлинеарных признаков:"))
                display(Markdown(f"{features_to_remove_corr[:10]}{'...' if len(features_to_remove_corr) > 10 else ''}"))
                X = X.drop(columns=features_to_remove_corr)
            else:
                display(Markdown("✅ Не обнаружено сильно коррелирующих признаков для удаления"))
        else:
            display(Markdown("ℹ️ Недостаточно данных для анализа мультиколлинеарности"))
    else:
        display(Markdown("ℹ️ Недостаточно числовых признаков для анализа мультиколлинеарности"))


    display(Markdown("### 🎯 Финальный отбор признаков"))

    final_features = X.columns.tolist()

    if len(final_features) > 50:
        numeric_final = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_final:
            feature_scores = {}
            for col in numeric_final:
                temp_df = pd.concat([X[col], y], axis=1).dropna()
                if len(temp_df) > 10:
                    try:
                        col_num = pd.to_numeric(temp_df[col], errors='coerce')
                        target_num = pd.to_numeric(temp_df.iloc[:, 1], errors='coerce')
                        valid_data = pd.concat([col_num, target_num], axis=1).dropna()
                        
                        if len(valid_data) > 10 and valid_data.iloc[:, 0].nunique() > 1:
                            corr = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
                            if not np.isnan(corr):
                                feature_scores[col] = abs(corr)
                    except:
                        continue
            
            if feature_scores:
                top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:50]
                final_features = [feat for feat, _ in top_features]
                
                missing_indicators = [col for col in X.columns if col.endswith('_missing')]
                final_features.extend([col for col in missing_indicators if col not in final_features])
                
                display(Markdown(f"### ✅ Отобрано {len(final_features)} наиболее значимых признаков"))
            else:
                final_features = X.columns[:50].tolist()
                display(Markdown("ℹ️ Не удалось оценить важность признаков - используем первые 50 признаков"))
        else:
            final_features = X.columns[:50].tolist()

    display(Markdown(f"### 📊 Финальные признаки для моделирования: {len(final_features)}"))
    display(Markdown(f"{final_features[:20]}{'...' if len(final_features) > 20 else ''}"))

    display(Markdown("## 📊 Шаг 4: Стратифицированное разделение данных"))

    def clean_column_names(df):
        """
        Очищает имена колонок от символов, которые не принимает XGBoost
        """
        new_columns = []
        for col in df.columns:
            clean_col = col.replace('[', '_').replace(']', '_').replace('<', '_lt_')
            clean_col = clean_col.replace('>', '_gt_').replace(' ', '_').replace(':', '_')
            clean_col = clean_col.replace('(', '_').replace(')', '_').replace(',', '_')
            clean_col = clean_col.replace('{', '_').replace('}', '_').replace('|', '_')
            clean_col = '_'.join(filter(None, clean_col.split('_')))
            clean_col = clean_col.strip('_')
            if clean_col and clean_col[0].isdigit():
                clean_col = 'f_' + clean_col
            new_columns.append(clean_col)
        
        df_clean = df.copy()
        df_clean.columns = new_columns
        return df_clean

    X_clean = clean_column_names(X)

    X_final = X_clean.copy()
    y_final = y.copy()

    total_missing = X_final.isnull().sum().sum()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_final, y_final,
                test_size=0.2,
                stratify=y_final,
                random_state=42
            )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=42
    )

    display(Markdown("### Пример данных после проведения всех манипуляций"))
    display(X_train)

    display(Markdown("### ✅ Успешное разделение данных!"))
    display(Markdown(f"**Размеры наборов:**"))
    display(Markdown(f"- Train: {len(X_train)} ({sum(y_train == 1)/len(y_train)*100:.1f}% класса 1)"))
    display(Markdown(f"- Validation: {len(X_val)} ({sum(y_val == 1)/len(y_val)*100:.1f}% класса 1)"))
    display(Markdown(f"- Test: {len(X_test)} ({sum(y_test == 1)/len(y_test)*100:.1f}% класса 1)"))

    strat_check = pd.DataFrame({
        'Набор': ['Train', 'Validation', 'Test'],
        'Процент класса 1': [
            sum(y_train == 1)/len(y_train)*100,
            sum(y_val == 1)/len(y_val)*100,
            sum(y_test == 1)/len(y_test)*100
        ]
    })
    display(Markdown("**Проверка стратификации (процент minority класса):**"))
    display(strat_check.style.background_gradient(cmap='Greens'))

    display(Markdown("## 📊 Базовая информация о датасете"))

    display(Markdown("### 💾 Информация о типах данных"))
    buffer = pd.io.common.StringIO()
    X.info(buf=buffer)
    info_str = buffer.getvalue()
    display(Markdown(f"```\n{info_str}\n```"))

    info_df = pd.DataFrame({
        'Column': X.columns,
        'Type': X.dtypes.astype(str),
        'First Value': X.iloc[0].values,
        'Last Value': X.iloc[-1].values,
        'Unique Values': X.nunique().values
    })

    display(info_df.style
            .set_caption("DataFrame Schema Overview")
            .format(precision=2)
            .background_gradient(cmap='Blues', subset=['Unique Values']))

    display(Markdown("### 📈 Описательная статистика числовых признаков"))
    display(X.describe().T.style.background_gradient(cmap='viridis'))

    display(Markdown("### 👀 Первые 5 строк данных"))
    display(X.head())

    display(Markdown("### 👀 Последние 5 строк данных"))
    display(X.tail())

    display(Markdown("### 🔍 Анализ уникальных значений (первые 10 столбцов)"))
    unique_counts = {}
    for col in X.columns[:10]:
        unique_counts[col] = {
            'unique_count': X[col].nunique(),
            'unique_values': X[col].unique()[:10].tolist(),
            'dtype': str(X[col].dtype)
        }

    unique_df = pd.DataFrame(unique_counts).T
    display(unique_df)

    display(Markdown("## 🤖 Шаг 2: Обучение моделей с поддержкой пропусков"))

    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        import lightgbm
        import time
        
        display(Markdown("✅ **Успешно импортированы модели с поддержкой пропусков:**"))
        display(Markdown("- XGBoost: автоматическая обработка NaN через параметр `missing`"))
        display(Markdown("- LightGBM: встроенная поддержка пропусков"))
        display(Markdown("- CatBoost: автоматическая обработка пропусков"))
        
    except ImportError as e:
        display(Markdown(f"❌ **Ошибка импорта моделей:** {e}"))
        display(Markdown("Установите необходимые библиотеки:"))
        display(Markdown("```bash"))
        display(Markdown("pip install xgboost lightgbm catboost"))
        display(Markdown("```"))
        display(Markdown("Пайплайн не может продолжить без этих моделей"))

    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    class_weight_dict = {0: 1, 1: scale_pos_weight}

    display(Markdown("### ⚖️ Параметры для работы с дисбалансом 97%/3%:"))
    display(Markdown(f"- **scale_pos_weight (XGBoost/LightGBM):** {scale_pos_weight:.1f}"))
    display(Markdown(f"- **class_weight (CatBoost):** {class_weight_dict}"))
    display(Markdown(f"- **Приоритетная метрика:** Recall для класса 1 (смерть)"))

    def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Обучает модель и оценивает её качество с фокусом на медицинские метрики
        Согласно официальной документации библиотек
        """
        start_time = time.time()
        
        display(Markdown(f"### 🚀 Обучение {model_name}..."))
        
        try:
            if model_name == 'XGBoost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                
            elif model_name == 'LightGBM':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)]  # Корректный синтаксис
                )
                
            elif model_name == 'CatBoost':
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            
            train_time = time.time() - start_time
            display(Markdown(f"✅ **{model_name} обучена за {train_time:.2f} секунд**"))
            
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            
            if hasattr(model, "predict_proba"):
                y_proba_train = model.predict_proba(X_train)[:, 1]
                y_proba_val = model.predict_proba(X_val)[:, 1]
                y_proba_test = model.predict_proba(X_test)[:, 1]
            else:
                y_proba_train = model.decision_function(X_train)
                y_proba_val = model.decision_function(X_val)
                y_proba_test = model.decision_function(X_test)
            
            results = {}
            
            for dataset_name, y_true, y_pred, y_proba in [
                ('Train', y_train, y_pred_train, y_proba_train),
                ('Validation', y_val, y_pred_val, y_proba_val),
                ('Test', y_test, y_pred_test, y_proba_test)
            ]:
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_proba)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                 
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall_curve, precision_curve)
                
                results[dataset_name] = {
                    'Recall (Sensitivity)': recall,
                    'Balanced Accuracy': balanced_accuracy,
                    'Precision': precision,
                    'F1-score': f1,
                    'ROC-AUC': roc_auc,
                    'PR-AUC': pr_auc
                }
            
            display(Markdown(f"### 📊 Результаты {model_name}:"))
            
            metrics_df = pd.DataFrame(results).T
            display(metrics_df.style.background_gradient(cmap='Blues', subset=['Recall (Sensitivity)', 'Balanced Accuracy', 'PR-AUC']))
            
            display(Markdown("### ⚠️ Анализ ошибок (крайне важно для медицины):"))
            
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel()

            display(Markdown("## 📊 Диагностика текущих результатов"))
            display(Markdown(f"### Confusion Matrix для {model_name}:"))
            cm_df = pd.DataFrame(cm, 
                        index=['Actual 0 (Выжил)', 'Actual 1 (Умер)'], 
                        columns=['Predicted 0', 'Predicted 1'])
            display(cm_df)
            
            display(Markdown(f"**Confusion Matrix (Test set):**"))
            display(Markdown(f"- True Negative (правильные выжившие): {tn}"))
            display(Markdown(f"- False Positive (ложные тревоги): {fp}"))
            display(Markdown(f"- False Negative (пропущенные смерти) ⚠️: {fn}"))
            display(Markdown(f"- True Positive (правильные предсказания смерти): {tp}"))
            
            if fn > 0:
                display(Markdown(f"❌ **КРИТИЧНО:** Модель пропустила {fn} смертельных случаев!"))
            if fp > 0:
                display(Markdown(f"⚠️ **Внимание:** {fp} ложных тревог могут привести к ненужным вмешательствам"))
            
            plt.figure(figsize=(10, 6))
            plt.plot(recall_curve, precision_curve, 'b-', linewidth=2, 
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall (Sensitivity)', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.show()
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Признак': X_train.columns,
                    'Важность': model.feature_importances_
                }).sort_values('Важность', ascending=False)
                
                display(Markdown(f"### 🌟 Топ-10 важных признаков ({model_name}):"))
                display(feature_importance.head(10).style.background_gradient(cmap='Greens'))
            
            return model, results
            
        except Exception as e:
            display(Markdown(f"❌ **Ошибка при обучении {model_name}:** {e}"))
            return None, None

    display(Markdown("## 🚀 Обучение baseline моделей"))

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        missing=np.nan,
        random_state=42,
        eval_metric='logloss',
        tree_method='auto'
    )

    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        class_weight=class_weight_dict,
        random_state=42,
        verbose=-1,
        early_stopping_rounds=50
    )

    cat_model = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.1,
        class_weights=[1, scale_pos_weight],
        random_state=42,
        verbose=False,
        eval_metric='Logloss',
        early_stopping_rounds=50
    )

    models_results = {}

    model_trained, results = train_and_evaluate_model(
        xgb_model, 'XGBoost', X_train, y_train, X_val, y_val, X_test, y_test
    )
    models_results['XGBoost'] = {'model': model_trained, 'results': results}

    model_trained, results = train_and_evaluate_model(
            lgb_model, 'LightGBM', X_train, y_train, X_val, y_val, X_test, y_test
        )
    models_results['LightGBM'] = {'model': model_trained, 'results': results}

    model_trained, results = train_and_evaluate_model(
        cat_model, 'CatBoost', X_train, y_train, X_val, y_val, X_test, y_test
    )
    if model_trained is not None:
        models_results['CatBoost'] = {'model': model_trained, 'results': results}

    display(Markdown("## 📊 Шаг 3: Сравнение моделей и выбор лучшей"))

    comparison_data = []
        
    for model_name, data in models_results.items():
        results = data['results']
        test_metrics = results['Test']
        
        comparison_data.append({
            'Модель': model_name,
            'Recall (Sensitivity)': test_metrics['Recall (Sensitivity)'],
            'Balanced Accuracy': test_metrics['Balanced Accuracy'],
            'Precision': test_metrics['Precision'],
            'F1-score': test_metrics['F1-score'],
            'PR-AUC': test_metrics['PR-AUC'],
            'ROC-AUC': test_metrics['ROC-AUC']
        })

    comparison_df = pd.DataFrame(comparison_data).sort_values('Balanced Accuracy', ascending=False)

    display(Markdown("### 🏆 Сравнение моделей на тестовом наборе:"))
    display(comparison_df.style.background_gradient(
        cmap='Blues', 
        subset=['Balanced Accuracy', 'PR-AUC']
    ).format({
        'Recall (Sensitivity)': '{:.3f}',
        'Balanced Accuracy': '{:.3f}',
        'Precision': '{:.3f}',
        'F1-score': '{:.3f}',
        'PR-AUC': '{:.3f}',
        'ROC-AUC': '{:.3f}'
    }))

    best_balanced_accuracy_model = comparison_df.iloc[0]['Модель']
    best_pr_auc_model = comparison_df.sort_values('PR-AUC', ascending=False).iloc[0]['Модель']

    display(Markdown("## 🎯 Рекомендации по выбору модели:"))

    if best_balanced_accuracy_model == best_pr_auc_model:
        display(Markdown(f"### ✅ **{best_balanced_accuracy_model} показала лучшие результаты по всем ключевым метрикам**"))
    else:
        display(Markdown(f"### 🔍 **Разные модели лидируют по разным метрикам:**"))
        display(Markdown(f"- **Лучший Balanced Accuracy:** {best_balanced_accuracy_model} - приоритет для медицины!"))
        display(Markdown(f"- **Лучший PR-AUC:** {best_pr_auc_model} - баланс precision/recall"))

    display(Markdown(f"## ⚙️ Оптимизация порога классификации для {best_balanced_accuracy_model}"))

    best_model_name = best_balanced_accuracy_model
    best_model = models_results[best_model_name]['model']

    if hasattr(best_model, "predict_proba"):
        y_proba_test = best_model.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = best_model.decision_function(X_test)

    thresholds = np.arange(0.1, 1, 0.05)
    best_threshold = 0.5
    best_balanced_accuracy = 0
    min_acceptable_precision = 0.1

    for threshold in thresholds:
        y_pred = (y_proba_test >= threshold).astype(int)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        if precision >= min_acceptable_precision and balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold

    display(Markdown(f"### 🎯 Оптимальный порог классификации: {best_threshold:.2f}"))
    display(Markdown(f"- **Balanced Accuracy при этом пороге:** {best_balanced_accuracy:.3f}"))
    display(Markdown(f"- **Precision при этом пороге:** {precision_score(y_test, (y_proba_test >= best_threshold).astype(int)):.3f}"))

    y_pred_optimal = (y_proba_test >= best_threshold).astype(int)
    final_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_optimal)
    final_recall = recall_score(y_test, y_pred_optimal)
    final_precision = precision_score(y_test, y_pred_optimal)
    final_f1 = f1_score(y_test, y_pred_optimal)

    display(Markdown("### 📊 Финальные метрики с оптимальным порогом:"))
    final_metrics_df = pd.DataFrame({
        'Метрика': ['Recall (Sensitivity)', 'Balanced Accuracy', 'Precision', 'F1-score'],
        'Значение': [final_recall, final_balanced_accuracy, final_precision, final_f1]
    })
    display(final_metrics_df.style.background_gradient(cmap='Greens', subset=['Значение']))

    best_model = models_results[best_model_name]['model']

    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]


    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    display(Markdown("## 📊 Диагностика текущих результатов"))
    display(Markdown(f"### Confusion Matrix для {best_model_name}:"))
    cm_df = pd.DataFrame(cm, 
                        index=['Actual 0 (Выжил)', 'Actual 1 (Умер)'], 
                        columns=['Predicted 0', 'Predicted 1'])
    display(cm_df)

    display(Markdown(f"### Критические метрики:"))
    display(Markdown(f"- **Recall (Sensitivity) для смерти:** {tp/(tp+fn):.3f}"))
    display(Markdown(f"- **Количество пропущенных смертей (FN):** {fn} из {tp+fn}"))
    display(Markdown(f"- **Precision для смерти:** {tp/(tp+fp):.3f}"))

    plt.figure(figsize=(12, 6))
    plt.hist(y_proba_test[y_test == 0], bins=50, alpha=0.5, label='Выжил (класс 0)', color='blue')
    plt.hist(y_proba_test[y_test == 1], bins=50, alpha=0.5, label='Умер (класс 1)', color='red')
    plt.axvline(x=0.5, color='k', linestyle='--', label='Порог 0.5')
    plt.axvline(x=best_threshold, color='k', linestyle='-.', label=f'Оптимальный порог {best_threshold}')
    plt.xlabel('Вероятность смерти')
    plt.ylabel('Количество наблюдений')
    plt.title('Распределение вероятностей по классам')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    if tp + fp == 0:
        display(Markdown("❌ **КРИТИЧЕСКАЯ ПРОБЛЕМА:** Модель вообще не предсказывает класс 1 (смерть)!"))
        display(Markdown("Это означает, что модель считает всех пациентов выжившими."))

    display(Markdown("## 🚀 Шаг 1: Гиперпараметрическая оптимизация"))

    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    display(Markdown(f"### ⚖️ Параметры дисбаланса: scale_pos_weight = {scale_pos_weight:.1f}"))

    best_models = {}

    display(Markdown("### 🤖 XGBoost оптимизация"))

    xgb_param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'min_child_weight': randint(1, 10),
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2]
    }

    xgb = XGBClassifier(
        missing=np.nan,
        random_state=42,
        eval_metric='logloss',
        tree_method='auto'
    )

    xgb_random = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_param_dist,
        n_iter=25,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    start_time = time.time()
    xgb_random.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_train_time = time.time() - start_time

    best_xgb = xgb_random.best_estimator_
    display(Markdown(f"✅ **XGBoost оптимизирован за {xgb_train_time:.2f} секунд**"))
    display(Markdown(f"**Лучшие параметры:**"))
    display(Markdown(f"```python\n{xgb_random.best_params_}\n```"))
    display(Markdown(f"**Лучший Balanced Accuracy на кросс-валидации:** {xgb_random.best_score_:.3f}"))

    display(Markdown("### 🤖 LightGBM оптимизация"))

    lgb_param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(15, 60),
        'min_child_samples': randint(5, 30),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'class_weight': [
            {0: 1, 1: scale_pos_weight},
            {0: 1, 1: scale_pos_weight * 1.5},
            {0: 1, 1: scale_pos_weight * 2}
        ]
    }

    lgb = LGBMClassifier(
        random_state=42,
        verbose=-1
    )

    lgb_random = RandomizedSearchCV(
        estimator=lgb,
        param_distributions=lgb_param_dist,
        n_iter=25,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    start_time = time.time()
    lgb_random.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lightgbm.early_stopping(20, verbose=False)])
    lgb_train_time = time.time() - start_time

    best_lgb = lgb_random.best_estimator_
    display(Markdown(f"✅ **LightGBM оптимизирован за {lgb_train_time:.2f} секунд**"))
    display(Markdown(f"**Лучшие параметры:**"))
    display(Markdown(f"```python\n{lgb_random.best_params_}\n```"))
    display(Markdown(f"**Лучший Balanced Accuracy на кросс-валидации:** {lgb_random.best_score_:.3f}"))

    display(Markdown("### 🤖 CatBoost оптимизация"))

    cat_param_dist = {
        'iterations': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'depth': randint(3, 10),
        'l2_leaf_reg': uniform(1, 10),
        'border_count': randint(32, 128),
        'class_weights': [
            [1, scale_pos_weight],
            [1, scale_pos_weight * 1.5],
            [1, scale_pos_weight * 2]
        ]
    }

    cat = CatBoostClassifier(
        random_state=42,
        verbose=False,
        eval_metric='F1',
        early_stopping_rounds=20
    )

    best_score = -1
    best_params = None
    best_model = None

    display(Markdown("🔍 **Запуск рандомизированного поиска для CatBoost...**"))
    start_time = time.time()

    for i in range(25):
        params = {
            'iterations': int(randint(50, 300).rvs()),
            'learning_rate': uniform(0.01, 0.3).rvs(),
            'depth': int(randint(3, 10).rvs()),
            'l2_leaf_reg': uniform(1, 10).rvs(),
            'border_count': int(randint(32, 128).rvs()),
            'class_weights': [1, scale_pos_weight * np.random.choice([1.0, 1.5, 2.0])]
        }
        
        model = CatBoostClassifier(
            **params,
            random_state=42,
            verbose=False,
            eval_metric='BalancedAccuracy',
            early_stopping_rounds=20
        )
        
        try:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
            
            y_pred = model.predict(X_val)
            score = balanced_accuracy_score(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
            
            if i % 5 == 0:
                display(Markdown(f"Итерация {i+1}/15: Balanced Accuracy = {score:.3f}"))
        
        except Exception as e:
            continue

    cat_train_time = time.time() - start_time
    best_cat = best_model

    display(Markdown(f"✅ **CatBoost оптимизирован за {cat_train_time:.2f} секунд**"))
    display(Markdown(f"**Лучшие параметры:**"))
    display(Markdown(f"```python\n{best_params}\n```"))
    display(Markdown(f"**Лучший Balanced Accuracy на validation:** {best_score:.3f}"))

    best_models = {
        'XGBoost': best_xgb,
        'LightGBM': best_lgb,
        'CatBoost': best_cat
    }

    display(Markdown("## 📊 Шаг 2: Оценка оптимизированных моделей"))

    results = []

    for name, model in best_models.items():
        display(Markdown(f"### 📈 Оценка {name}"))
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        results.append({
            'Модель': name,
            'Recall': recall,
            'Balanced Accuracy': balanced_accuracy,
            'Precision': precision,
            'F1-score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })
        
        display(Markdown(f"**Метрики на тестовом наборе:**"))
        metrics_df = pd.DataFrame({
            'Метрика': ['Recall', 'Balanced Accuracy', 'Precision', 'F1-score', 'ROC-AUC', 'PR-AUC'],
            'Значение': [recall, balanced_accuracy, precision, f1, roc_auc, pr_auc]
        })
        display(metrics_df.style.background_gradient(cmap='Blues', subset=['Значение']))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {name}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(recall_curve, precision_curve, 'g-', linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {name}')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    results_df = pd.DataFrame(results)
    display(Markdown("## 🏆 Сравнение оптимизированных моделей:"))
    display(results_df.sort_values('Balanced Accuracy', ascending=False)
        .style.background_gradient(cmap='Blues', subset=['F1-score', 'ROC-AUC', 'Balanced Accuracy'])
        .background_gradient(cmap='Greens', subset=['Recall'])
        .background_gradient(cmap='Oranges', subset=['Precision']))

    best_model_name = results_df.sort_values('Balanced Accuracy', ascending=False).iloc[0]['Модель']
    best_model = best_models[best_model_name]

    display(Markdown(f"## 🎯 Лучшая модель по Balanced Accuracy: **{best_model_name}**"))