import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_recall_curve, auc, confusion_matrix, recall_score, precision_score, balanced_accuracy_score)
from scipy.stats import uniform, randint
import lightgbm
import time
import os
from typing import Optional, Dict, Any, Tuple, List

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️ Модели не импортированы. Установите: pip install xgboost lightgbm catboost")

class DataAnalyzer:
    """Модуль для разведочного анализа данных (EDA)"""
    
    def __init__(self, target_variable: str = 'Смерть', target_mapping: Dict = {'Да': 1, 'Нет': 0}):
        self.target_variable = target_variable
        self.target_mapping = target_mapping
        self.df = None
        self.analysis_results = {}
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        warnings.filterwarnings('ignore')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)
    
    def load_data(self, df: pd.DataFrame) -> 'DataAnalyzer':
        """Загрузка данных с копированием для безопасности"""
        self.df = df.copy()
        return self
    
    def run_full_eda(self, output_dir: Optional[str] = None) -> 'DataAnalyzer':
        """Полный EDA с сохранением результатов"""
        self.show_basic_info()
        self.analyze_class_imbalance()
        self.analyze_missing_values()
        self.detect_data_leaks()
        self.analyze_multicollinearity()
        self.analyze_feature_distributions()
        
        if output_dir:
            self.save_eda_report(output_dir)
        
        return self
    
    def show_basic_info(self) -> 'DataAnalyzer':
        """Базовая информация о датасете"""
        if self.df is None:
            raise ValueError("Данные не загружены. Вызовите load_data() сначала.")
        
        display(Markdown("## 📊 Базовая информация о датасете"))
        
        display(Markdown("### 💾 Информация о типах данных"))
        buffer = pd.io.common.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        display(Markdown(f"```\n{info_str}\n```"))
        
        info_df = pd.DataFrame({
            'Column': self.df.columns,
            'Type': self.df.dtypes.astype(str),
            'First Value': self.df.iloc[0].values,
            'Last Value': self.df.iloc[-1].values,
            'Unique Values': self.df.nunique().values
        })
        display(info_df.style
                .set_caption("DataFrame Schema Overview")
                .format(precision=2)
                .background_gradient(cmap='Blues', subset=['Unique Values']))
        
        display(Markdown("### 📈 Описательная статистика числовых признаков"))
        display(self.df.describe().T.style.background_gradient(cmap='viridis'))
        
        display(Markdown("### 👀 Первые 5 строк данных"))
        display(self.df.head())
        display(Markdown("### 👀 Последние 5 строк данных"))
        display(self.df.tail())
        
        display(Markdown("### 🔍 Анализ уникальных значений (первые 10 столбцов)"))
        unique_counts = {}
        for col in self.df.columns[:10]:
            unique_counts[col] = {
                'unique_count': self.df[col].nunique(),
                'unique_values': self.df[col].unique()[:10].tolist(),
                'dtype': str(self.df[col].dtype)
            }
        unique_df = pd.DataFrame(unique_counts).T
        display(unique_df)
        
        return self
    
    def analyze_class_imbalance(self) -> 'DataAnalyzer':
        """Анализ дисбаланса классов"""
        if self.target_variable not in self.df.columns:
            raise ValueError(f"Целевая переменная '{self.target_variable}' не найдена в данных")
        
        if self.df[self.target_variable].dtype == 'object':
            self.df[self.target_variable] = self.df[self.target_variable].map(self.target_mapping)
        
        class_counts = self.df[self.target_variable].value_counts()
        class_percentages = (class_counts / len(self.df)) * 100
        
        display(Markdown("## ⚖️ Анализ дисбаланса классов"))
        display(Markdown(f"### Целевая переменная: `{self.target_variable}`"))
        
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
                   f'{height:,} ({height/len(self.df)*100:.1f}%)',
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
        
        self.analysis_results['class_imbalance'] = {
            'counts': class_counts.to_dict(),
            'percentages': class_percentages.to_dict(),
            'ratio': imbalance_ratio
        }
        
        return self
    
    def analyze_missing_values(self) -> 'DataAnalyzer':
        """Анализ пропущенных значений"""
        display(Markdown("## 🕳️ Анализ пропущенных значений"))
        
        missing_percentages = (self.df.isnull().sum() / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Столбец': missing_percentages.index,
            'Процент пропусков': missing_percentages.values
        }).sort_values('Процент пропусков', ascending=False)
        missing_df = missing_df[missing_df['Процент пропусков'] > 0]
        
        display(Markdown(f"### 📊 Общая статистика по пропускам:"))
        display(Markdown(f"- Всего столбцов с пропусками: {len(missing_df)} из {len(self.df.columns)}"))
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
        
        if self.target_variable in self.df.columns:
            display(Markdown("### 🎯 Анализ пропусков в контексте целевой переменной"))
            key_missing_cols = missing_df.head(5)['Столбец'].tolist()
            for col in key_missing_cols:
                if col != self.target_variable:
                    display(Markdown(f"#### Столбец: `{col}`"))
                    missing_analysis = pd.crosstab(
                        self.df[self.target_variable],
                        self.df[col].isnull().map({True: 'Пропуск', False: 'Значение'}),
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
        
        self.analysis_results['missing_values'] = missing_df.to_dict()
        return self
    
    def detect_data_leaks(self) -> 'DataAnalyzer':
        """Поиск потенциальных утечек данных"""
        display(Markdown("## 🔍 Поиск утечек данных (data leaks)"))
        
        df_analysis = self.df.copy()
        cols_to_drop = df_analysis.columns[df_analysis.isnull().mean() == 1.0]
        if len(cols_to_drop) > 0:
            df_analysis = df_analysis.drop(columns=cols_to_drop)
        
        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_variable in numeric_cols:
            numeric_cols.remove(self.target_variable)
        
        display(Markdown(f"### Числовые признаки для анализа: {len(numeric_cols)} из {len(df_analysis.columns)}"))
        
        if self.target_variable in df_analysis.columns and len(numeric_cols) > 0:
            target_numeric = df_analysis[self.target_variable]
            correlations = {}
            for col in numeric_cols:
                temp_df = df_analysis[[col, self.target_variable]].dropna()
                if len(temp_df) > 10:
                    try:
                        corr = temp_df[col].corr(temp_df[self.target_variable])
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
                display(corr_df.head(20).style.background_gradient(
                    cmap='coolwarm', subset=['Корреляция с target'], vmin=-1, vmax=1))
                
                if not potential_leaks.empty:
                    display(Markdown(f"### ⚠️ **ПОТЕНЦИАЛЬНЫЕ УТЕЧКИ ДАННЫХ** (|корреляция| >= {high_corr_threshold}):"))
                    display(potential_leaks.style.background_gradient(cmap='Reds', subset=['Корреляция с target']))
                
                plt.figure(figsize=(14, 8))
                top_corr = corr_df.head(25)
                bars = plt.barh(top_corr['Признак'], top_corr['Корреляция с target'],
                               color=['#ff4444' if abs(x) >= high_corr_threshold else '#4488ff' 
                                      for x in top_corr['Корреляция с target']])
                plt.axvline(x=high_corr_threshold, color='r', linestyle='--', alpha=0.7, 
                           label=f'Порог утечки ({high_corr_threshold})')
                plt.axvline(x=-high_corr_threshold, color='r', linestyle='--', alpha=0.7)
                plt.title(f'Топ-25 корреляций с целевой переменной "{self.target_variable}"', 
                         fontsize=15, fontweight='bold')
                plt.xlabel('Корреляция с target', fontsize=12)
                plt.ylabel('Признаки', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
                
                self.analysis_results['potential_leaks'] = potential_leaks.to_dict()
        
        return self
    
    def analyze_multicollinearity(self, threshold: float = 0.95) -> 'DataAnalyzer':
        """Анализ мультиколлинеарности признаков"""
        display(Markdown("## 📐 Анализ мультиколлинеарности признаков"))
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_variable in numeric_cols:
            numeric_cols.remove(self.target_variable)
        
        if len(numeric_cols) > 1:
            top_features = numeric_cols[:50] if len(numeric_cols) > 50 else numeric_cols
            
            corr_matrix = self.df[top_features].corr().abs()
            
            plt.figure(figsize=(16, 14))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                       center=0, square=True, linewidths=0.5,
                       cbar_kws={"shrink": .5}, vmin=0, vmax=1)
            plt.title('Матрица корреляций между признаками', fontsize=16, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.show()
            
            high_corr_pairs = []
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
                self.analysis_results['multicollinearity'] = high_corr_df.to_dict()
            else:
                display(Markdown("✅ Не обнаружено признаков с экстремально высокой корреляцией между собой"))
        
        return self
    
    def analyze_feature_distributions(self, n_top_features: int = 10) -> 'DataAnalyzer':
        """Анализ распределений признаков по классам target"""
        display(Markdown("## 📈 Анализ распределений признаков по классам target"))
        
        if self.target_variable not in self.df.columns:
            display(Markdown("⚠️ Целевая переменная не найдена, пропускаем анализ распределений"))
            return self
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_variable in numeric_cols:
            numeric_cols.remove(self.target_variable)
        
        if not numeric_cols:
            display(Markdown("⚠️ Нет числовых признаков для анализа распределений"))
            return self
        
        correlations = {}
        for col in numeric_cols:
            temp_df = self.df[[col, self.target_variable]].dropna()
            if len(temp_df) > 10:
                try:
                    corr = temp_df[col].corr(temp_df[self.target_variable])
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except:
                    continue
        
        if not correlations:
            display(Markdown("⚠️ Не удалось вычислить корреляции для выбора признаков"))
            top_features = numeric_cols[:n_top_features]
        else:
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:n_top_features]
            top_features = [feat for feat, _ in top_features]
        
        if 'potential_leaks' in self.analysis_results:
            leak_features = list(self.analysis_results['potential_leaks']['Признак'].keys())
            top_features = [f for f in top_features if f not in leak_features]
        
        if top_features:
            display(Markdown(f"### Распределения топ-{len(top_features)} признаков по классам target:"))
            target_classes = self.df[self.target_variable].dropna().unique()
            target_classes = target_classes[~pd.isna(target_classes)]
            
            n_cols = 3
            n_rows = (len(top_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                   squeeze=False, tight_layout=True)
            
            for i, feature in enumerate(top_features):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                for class_val in target_classes:
                    class_data = self.df[self.df[self.target_variable] == class_val][feature]
                    class_data = class_data.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(class_data) > 10:
                        sns.kdeplot(class_data, ax=ax, label=f'Class {class_val}',
                                   alpha=0.7, linewidth=2)
                ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Значение', fontsize=10)
                ax.set_ylabel('Плотность', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
            
            for i in range(len(top_features), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.suptitle('Распределения признаков по классам target', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        return self
    
    def save_eda_report(self, output_dir: str) -> 'DataAnalyzer':
        """Сохранение результатов EDA в файлы"""
        os.makedirs(output_dir, exist_ok=True)
        
        if 'class_imbalance' in self.analysis_results:
            pd.DataFrame(self.analysis_results['class_imbalance']['counts'], index=[0]).to_csv(
                os.path.join(output_dir, 'class_imbalance.csv'), index=False)
        
        if 'missing_values' in self.analysis_results:
            pd.DataFrame(self.analysis_results['missing_values']).to_csv(
                os.path.join(output_dir, 'missing_values.csv'), index=False)
        
        display(Markdown(f"✅ Отчёты EDA сохранены в: {output_dir}"))
        return self
    
    def get_data(self) -> pd.DataFrame:
        """Получить обработанные данные"""
        return self.df.copy() if self.df is not None else None


class DataPreprocessor:
    """Модуль для предобработки данных"""
    
    def __init__(self, target_variable: str = 'Смерть'):
        self.target_variable = target_variable
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.preprocessing_info = {}
    
    def load_data(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Загрузка данных"""
        self.df = df.copy()
        return self
    
    def prepare_target(self) -> 'DataPreprocessor':
        """Подготовка целевой переменной"""
        if self.target_variable not in self.df.columns:
            raise ValueError(f"Целевая переменная '{self.target_variable}' не найдена")
        
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.target_variable])
        removed = initial_count - len(self.df)
        
        if removed > 0:
            display(Markdown(f"### 🗑️ Удалено {removed} строк с пропусками в целевой переменной"))
        
        if self.df[self.target_variable].dtype == 'object':
            unique_vals = self.df[self.target_variable].unique()
            if set(unique_vals) == {'Да', 'Нет'}:
                self.df[self.target_variable] = self.df[self.target_variable].map({'Да': 1, 'Нет': 0})
                display(Markdown("✅ Целевая переменная преобразована: 'Да'→1, 'Нет'→0"))
            else:
                self.df[self.target_variable] = self.df[self.target_variable].astype('category').cat.codes
                display(Markdown("⚠️ Целевая переменная закодирована как категориальная"))
        
        self.y = self.df[self.target_variable].astype(int)
        self.X = self.df.drop(columns=[self.target_variable])
        
        class_counts = self.y.value_counts()
        display(Markdown("## ⚖️ Финальный дисбаланс классов:"))
        balance_df = pd.DataFrame({
            'Класс': class_counts.index,
            'Количество': class_counts.values,
            'Процент': (class_counts / len(self.y) * 100).values
        })
        display(balance_df.style.background_gradient(cmap='Reds'))
        
        if len(class_counts) < 2:
            raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: Остался только один класс!")
        elif min(class_counts) < 10:
            display(Markdown(f"⚠️ **ОПАСНЫЙ ДИСБАЛАНС: В minority классе всего {min(class_counts)} наблюдений!**"))
        
        self.preprocessing_info['class_counts'] = class_counts.to_dict()
        self.preprocessing_info['scale_pos_weight'] = len(self.y[self.y == 0]) / max(1, len(self.y[self.y == 1]))
        
        return self
    
    def handle_missing_values(self, high_missing_threshold: float = 70.0) -> 'DataPreprocessor':
        """Обработка пропущенных значений"""
        display(Markdown("## 🕳️ Шаг 2: Стратегическая обработка пропусков"))
        
        missing_percentages = (self.X.isnull().sum() / len(self.X)) * 100
        cols_to_drop = missing_percentages[missing_percentages > high_missing_threshold].index.tolist()
        
        if cols_to_drop:
            display(Markdown(f"### 🗑️ Удаляем {len(cols_to_drop)} признаков с >{high_missing_threshold}% пропусков:"))
            display(Markdown(f"{cols_to_drop[:10]}{'...' if len(cols_to_drop) > 10 else ''}"))
            self.X = self.X.drop(columns=cols_to_drop)
        else:
            display(Markdown("✅ Нет признаков для удаления по критерию >70% пропусков"))
        
        empty_cols = self.X.columns[self.X.isnull().all()].tolist()
        if empty_cols:
            self.X = self.X.drop(columns=empty_cols)
        
        constant_cols = self.X.columns[self.X.nunique() == 1].tolist()
        if constant_cols:
            display(Markdown(f"Удаляем константные признаки: {constant_cols[:5]}{'...' if len(constant_cols) > 5 else ''}"))
            self.X = self.X.drop(columns=constant_cols)
        
        self.preprocessing_info['missing_values_handled'] = True
        return self
    
    def encode_categorical_features(self) -> 'DataPreprocessor':
        """Кодирование категориальных признаков"""
        display(Markdown("## 🔤 Кодирование категориальных признаков"))
        
        X_encoded = self.X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        display(Markdown(f"### 📊 Распределение типов признаков:"))
        display(Markdown(f"- **Числовых признаков:** {len(X_encoded.select_dtypes(include=[np.number]).columns)}"))
        display(Markdown(f"- **Категориальных признаков:** {len(categorical_cols)}"))
        display(Markdown(f"- **Всего признаков:** {len(X_encoded.columns)}"))
        
        if not categorical_cols:
            display(Markdown("✅ Нет категориальных признаков для обработки"))
            self.X = X_encoded
            return self
        
        display(Markdown("### 🔍 Примеры категориальных признаков:"))
        for col in categorical_cols[:5]:
            unique_vals = X_encoded[col].dropna().unique()[:5]
            display(Markdown(f"- `{col}`: примеры значений {unique_vals}{'...' if len(unique_vals) > 5 else ''}"))
        
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
        self.X = X_encoded
        return self
    
    def remove_leaky_features(self) -> 'DataPreprocessor':
        """Удаление признаков, присутствующих только в одном классе (скрытые утечки)"""
        display(Markdown("## 🔍 Поиск признаков, присутствующих только в одном классе"))
        
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        one_class_features = []
        
        for col in numeric_cols:
            class_0_data = self.X[self.y == 0][col]
            class_1_data = self.X[self.y == 1][col]
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
            self.X = self.X.drop(columns=features_to_drop)
            self.preprocessing_info['leaky_features_removed'] = features_to_drop
        else:
            display(Markdown("✅ Не обнаружено признаков, присутствующих только в одном классе"))
        
        return self
    
    def handle_multicollinearity(self, threshold: float = 0.85) -> 'DataPreprocessor':
        """Устранение мультиколлинеарности"""
        display(Markdown("## 📐 Устранение мультиколлинеарности"))
        
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        features_to_remove = []
        
        if len(numeric_cols) > 5:
            corr_matrix = self.X[numeric_cols].corr().abs()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > threshold:
                        feature1 = corr_matrix.columns[i]
                        feature2 = corr_matrix.columns[j]
                        
                        temp_df1 = pd.concat([self.X[feature1], self.y], axis=1).dropna()
                        temp_df2 = pd.concat([self.X[feature2], self.y], axis=1).dropna()
                        
                        if len(temp_df1) > 10 and len(temp_df2) > 10:
                            try:
                                corr1 = temp_df1.iloc[:, 0].corr(temp_df1.iloc[:, 1])
                                corr2 = temp_df2.iloc[:, 0].corr(temp_df2.iloc[:, 1])
                                feature_to_remove = feature1 if abs(corr1) < abs(corr2) else feature2
                                
                                if feature_to_remove not in features_to_remove:
                                    features_to_remove.append(feature_to_remove)
                            except:
                                continue
            
            if features_to_remove:
                display(Markdown(f"### 🗑️ Удаляем {len(features_to_remove)} мультиколлинеарных признаков:"))
                display(Markdown(f"{features_to_remove[:10]}{'...' if len(features_to_remove) > 10 else ''}"))
                self.X = self.X.drop(columns=features_to_remove)
                self.preprocessing_info['multicollinear_features_removed'] = features_to_remove
            else:
                display(Markdown("✅ Не обнаружено сильно коррелирующих признаков для удаления"))
        
        return self
    
    def select_features(self, max_features: int = 50) -> 'DataPreprocessor':
        """Финальный отбор признаков"""
        display(Markdown("## 🎯 Финальный отбор признаков"))
        
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) <= max_features:
            display(Markdown(f"✅ Признаков ({len(numeric_cols)}) меньше лимита ({max_features}), отбор не требуется"))
            self.feature_names = self.X.columns.tolist()
            return self
        
        feature_scores = {}
        for col in numeric_cols:
            temp_df = pd.concat([self.X[col], self.y], axis=1).dropna()
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
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
            final_features = [feat for feat, _ in top_features]
            
            missing_indicators = [col for col in self.X.columns if col.endswith('_missing')]
            final_features.extend([col for col in missing_indicators if col not in final_features])
            
            display(Markdown(f"### ✅ Отобрано {len(final_features)} наиболее значимых признаков"))
            self.X = self.X[final_features]
            self.feature_names = final_features
        else:
            final_features = self.X.columns[:max_features].tolist()
            display(Markdown("ℹ️ Не удалось оценить важность признаков - используем первые признаки"))
            self.X = self.X[final_features]
            self.feature_names = final_features
        
        display(Markdown(f"### 📊 Финальные признаки для моделирования: {len(self.feature_names)}"))
        display(Markdown(f"{self.feature_names[:20]}{'...' if len(self.feature_names) > 20 else ''}"))
        
        self.preprocessing_info['selected_features'] = self.feature_names
        return self
    
    def clean_column_names(self) -> 'DataPreprocessor':
        """Очистка имён колонок для совместимости с моделями"""
        def clean_name(col):
            clean_col = (str(col)
                .replace('[', '_').replace(']', '_').replace('<', '_lt_')
                .replace('>', '_gt_').replace(' ', '_').replace(':', '_')
                .replace('(', '_').replace(')', '_').replace(',', '_')
                .replace('{', '_').replace('}', '_').replace('|', '_')
                .replace('=', '_eq_').replace('!', '_not_'))
            clean_col = '_'.join(filter(None, clean_col.split('_')))
            clean_col = clean_col.strip('_')
            if clean_col and clean_col[0].isdigit():
                clean_col = 'f_' + clean_col
            return clean_col
        
        self.X.columns = [clean_name(col) for col in self.X.columns]
        self.feature_names = self.X.columns.tolist()
        return self
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42) -> 'DataPreprocessor':
        """Стратифицированное разделение данных"""
        display(Markdown("## 📊 Стратифицированное разделение данных"))
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size,
            stratify=y_train_val,
            random_state=random_state
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
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
        
        self.preprocessing_info['split_sizes'] = {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        }
        return self
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Получить обработанные данные"""
        if self.X_train is None:
            raise ValueError("Данные не разделены. Вызовите split_data() сначала.")
        return (self.X_train.copy(), self.y_train.copy(),
                self.X_val.copy(), self.y_val.copy(),
                self.X_test.copy(), self.y_test.copy())
    
    def get_scale_pos_weight(self) -> float:
        """Получить вес для балансировки классов"""
        return self.preprocessing_info.get('scale_pos_weight', 1.0)
    
    def save_preprocessed_data(self, output_dir: str) -> 'DataPreprocessor':
        """Сохранение предобработанных данных"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.X_train is not None:
            self.X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
            self.y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
            self.X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
            self.y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
            self.X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
            self.y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
            
            pd.DataFrame([self.preprocessing_info]).to_json(
                os.path.join(output_dir, 'preprocessing_info.json'), orient='records')
            
            display(Markdown(f"✅ Предобработанные данные сохранены в: {output_dir}"))
        
        return self


class ModelTrainer:
    """Модуль для обучения и оценки моделей"""
    
    def __init__(self, target_variable: str = 'Смерть'):
        self.target_variable = target_variable
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_baseline_models(self, 
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            scale_pos_weight: float = 1.0) -> 'ModelTrainer':
        """Обучение baseline моделей"""
        if not MODELS_AVAILABLE:
            raise ImportError("Модели не доступны. Установите: pip install xgboost lightgbm catboost")
        
        display(Markdown("## 🤖 Обучение baseline моделей"))
        display(Markdown(f"### ⚖️ Параметры для работы с дисбалансом:"))
        display(Markdown(f"- **scale_pos_weight:** {scale_pos_weight:.1f}"))
        
        class_weight_dict = {0: 1, 1: scale_pos_weight}
        
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
        self._train_and_evaluate(xgb_model, 'XGBoost', X_train, y_train, X_val, y_val, X_test, y_test)
        
        lgb_model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            class_weight=class_weight_dict,
            random_state=42,
            verbose=-1
        )
        self._train_and_evaluate(lgb_model, 'LightGBM', X_train, y_train, X_val, y_val, X_test, y_test)
        
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
        self._train_and_evaluate(cat_model, 'CatBoost', X_train, y_train, X_val, y_val, X_test, y_test)
        
        return self
    
    def _train_and_evaluate(self, model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """Внутренний метод для обучения и оценки одной модели"""
        start_time = time.time()
        display(Markdown(f"### 🚀 Обучение {model_name}..."))
        
        try:
            if model_name == 'XGBoost':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == 'LightGBM':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)])
            elif model_name == 'CatBoost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val),
                         early_stopping_rounds=50, verbose=False)
            
            train_time = time.time() - start_time
            display(Markdown(f"✅ **{model_name} обучена за {train_time:.2f} секунд**"))
            
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            metrics = self._calculate_metrics(y_test, y_pred_test, y_proba_test)
            
            self.models[model_name] = model
            self.results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred_test,
                'probabilities': y_proba_test,
                'training_time': train_time
            }
            
            self._display_results(model_name, metrics, y_test, y_pred_test, y_proba_test)
            
        except Exception as e:
            display(Markdown(f"❌ **Ошибка при обучении {model_name}:** {e}"))
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Расчёт метрик качества"""
        metrics = {
            'Recall (Sensitivity)': recall_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else None
        }
        
        if y_proba is not None:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            metrics['PR-AUC'] = auc(recall_curve, precision_curve)
        else:
            metrics['PR-AUC'] = None
        
        return metrics
    
    def _display_results(self, model_name, metrics, y_true, y_pred, y_proba):
        """Отображение результатов обучения"""
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Значение']
        display(Markdown(f"### 📊 Результаты {model_name}:"))
        display(metrics_df.style.background_gradient(cmap='Blues'))
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cm_df = pd.DataFrame(cm,
                           index=['Actual 0 (Выжил)', 'Actual 1 (Умер)'],
                           columns=['Predicted 0', 'Predicted 1'])
        display(Markdown(f"### Confusion Matrix для {model_name}:"))
        display(cm_df)
        
        display(Markdown(f"**Критические метрики:**"))
        display(Markdown(f"- **Recall (Sensitivity) для смерти:** {tp/(tp+fn):.3f}"))
        display(Markdown(f"- **Количество пропущенных смертей (FN):** {fn} из {tp+fn}"))
        display(Markdown(f"- **Precision для смерти:** {tp/(tp+fp):.3f}"))
        
        if y_proba is not None:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            plt.figure(figsize=(10, 6))
            plt.plot(recall_curve, precision_curve, 'b-', linewidth=2,
                    label=f'PR curve (AUC = {metrics["PR-AUC"]:.3f})')
            plt.xlabel('Recall (Sensitivity)', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.show()
        
        if hasattr(self.models[model_name], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Признак': self.models[model_name].feature_names_in_ 
                          if hasattr(self.models[model_name], 'feature_names_in_') 
                          else X_train.columns,
                'Важность': self.models[model_name].feature_importances_
            }).sort_values('Важность', ascending=False)
            display(Markdown(f"### 🌟 Топ-10 важных признаков ({model_name}):"))
            display(feature_importance.head(10).style.background_gradient(cmap='Greens'))
    
    def compare_models(self) -> pd.DataFrame:
        """Сравнение моделей"""
        if not self.results:
            raise ValueError("Нет обученных моделей для сравнения")
        
        comparison_data = []
        for model_name, data in self.results.items():
            metrics = data['metrics']
            comparison_data.append({
                'Модель': model_name,
                'Recall (Sensitivity)': metrics['Recall (Sensitivity)'],
                'Balanced Accuracy': metrics['Balanced Accuracy'],
                'Precision': metrics['Precision'],
                'F1-score': metrics['F1-score'],
                'PR-AUC': metrics['PR-AUC'],
                'ROC-AUC': metrics['ROC-AUC'],
                'Training Time (s)': data['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Balanced Accuracy', ascending=False)
        
        display(Markdown("## 🏆 Сравнение моделей на тестовом наборе:"))
        display(comparison_df.style.background_gradient(
            cmap='Blues', subset=['Balanced Accuracy', 'PR-AUC', 'ROC-AUC']
        ).format({
            'Recall (Sensitivity)': '{:.3f}',
            'Balanced Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'F1-score': '{:.3f}',
            'PR-AUC': '{:.3f}',
            'ROC-AUC': '{:.3f}',
            'Training Time (s)': '{:.2f}'
        }))
        
        best_model = comparison_df.iloc[0]['Модель']
        display(Markdown(f"### ✅ **Рекомендуемая модель:** {best_model} (лучший Balanced Accuracy)"))
        
        self.best_model_name = best_model
        self.best_model = self.models[best_model]
        
        return comparison_df
    
    def optimize_threshold(self, model_name: Optional[str] = None, 
                          min_precision: float = 0.1) -> Tuple[float, Dict]:
        """Оптимизация порога классификации"""
        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("Не указана модель и нет лучшей модели по умолчанию")
            model_name = self.best_model_name
        
        if model_name not in self.results:
            raise ValueError(f"Модель '{model_name}' не обучена")
        
        model = self.models[model_name]
        y_test = self.results[model_name].get('y_true')
        y_proba = self.results[model_name]['probabilities']
        
        display(Markdown(f"## ⚙️ Оптимизация порога классификации для {model_name}"))
        
        thresholds = np.arange(0.05, 1.0, 0.05)
        best_threshold = 0.5
        best_metric = 0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            
            if precision >= min_precision and balanced_accuracy > best_metric:
                best_metric = balanced_accuracy
                best_threshold = threshold
        
        display(Markdown(f"### 🎯 Оптимальный порог: {best_threshold:.2f}"))
        display(Markdown(f"- **Balanced Accuracy:** {best_metric:.3f}"))
        display(Markdown(f"- **Precision:** {precision_score(y_test, (y_proba >= best_threshold).astype(int)):.3f}"))
        
        return best_threshold, {'threshold': best_threshold, 'balanced_accuracy': best_metric}
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Получить лучшую модель"""
        if self.best_model is None:
            self.compare_models()
        return self.best_model_name, self.best_model


class HyperparameterOptimizer:
    """Модуль для гиперпараметрической оптимизации"""
    
    def __init__(self, target_variable: str = 'Смерть'):
        self.target_variable = target_variable
        self.best_models = {}
        self.best_params = {}
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, scale_pos_weight: float,
                        n_iter: int = 25, cv_folds: int = 3) -> Tuple[Any, Dict]:
        """Оптимизация XGBoost"""
        if not MODELS_AVAILABLE:
            raise ImportError("XGBoost не доступен")
        
        display(Markdown("### 🤖 XGBoost оптимизация"))
        
        param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'min_child_weight': randint(1, 10),
            'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2]
        }
        
        model = XGBClassifier(
            missing=np.nan,
            random_state=42,
            eval_metric='logloss',
            tree_method='auto'
        )
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_time = time.time() - start_time
        
        display(Markdown(f"✅ **XGBoost оптимизирован за {train_time:.2f} секунд**"))
        display(Markdown(f"**Лучшие параметры:**"))
        display(Markdown(f"```python\n{random_search.best_params_}\n```"))
        display(Markdown(f"**Лучший Balanced Accuracy на кросс-валидации:** {random_search.best_score_:.3f}"))
        
        self.best_models['XGBoost'] = random_search.best_estimator_
        self.best_params['XGBoost'] = random_search.best_params_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, scale_pos_weight: float,
                         n_iter: int = 25, cv_folds: int = 3) -> Tuple[Any, Dict]:
        """Оптимизация LightGBM"""
        if not MODELS_AVAILABLE:
            raise ImportError("LightGBM не доступен")
        
        display(Markdown("### 🤖 LightGBM оптимизация"))
        
        param_dist = {
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
        
        model = LGBMClassifier(random_state=42, verbose=-1)
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lightgbm.early_stopping(20, verbose=False)])
        train_time = time.time() - start_time
        
        display(Markdown(f"✅ **LightGBM оптимизирован за {train_time:.2f} секунд**"))
        display(Markdown(f"**Лучшие параметры:**"))
        display(Markdown(f"```python\n{random_search.best_params_}\n```"))
        display(Markdown(f"**Лучший Balanced Accuracy на кросс-валидации:** {random_search.best_score_:.3f}"))
        
        self.best_models['LightGBM'] = random_search.best_estimator_
        self.best_params['LightGBM'] = random_search.best_params_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val, scale_pos_weight: float,
                         n_iter: int = 15) -> Tuple[Any, Dict]:
        """Оптимизация CatBoost (кастомный поиск из-за специфики API)"""
        if not MODELS_AVAILABLE:
            raise ImportError("CatBoost не доступен")
        
        display(Markdown("### 🤖 CatBoost оптимизация"))
        
        best_score = -1
        best_params = None
        best_model = None
        
        start_time = time.time()
        for i in range(n_iter):
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
                model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
                y_pred = model.predict(X_val)
                score = balanced_accuracy_score(y_val, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                
                if i % 5 == 0 and i > 0:
                    display(Markdown(f"Итерация {i+1}/{n_iter}: Balanced Accuracy = {score:.3f}"))
            except Exception as e:
                continue
        
        train_time = time.time() - start_time
        
        display(Markdown(f"✅ **CatBoost оптимизирован за {train_time:.2f} секунд**"))
        display(Markdown(f"**Лучшие параметры:**"))
        display(Markdown(f"```python\n{best_params}\n```"))
        display(Markdown(f"**Лучший Balanced Accuracy на validation:** {best_score:.3f}"))
        
        self.best_models['CatBoost'] = best_model
        self.best_params['CatBoost'] = best_params
        
        return best_model, best_params
    
    def get_best_models(self) -> Dict[str, Any]:
        """Получить все оптимизированные модели"""
        return self.best_models


class MLPipeline:
    """Главный класс для оркестрации всего пайплайна"""
    
    def __init__(self, target_variable: str = 'Смерть', 
                 target_mapping: Dict = {'Да': 1, 'Нет': 0}):
        self.target_variable = target_variable
        self.target_mapping = target_mapping
        self.df = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.scale_pos_weight = None
        
        # Модули
        self.analyzer = DataAnalyzer(target_variable, target_mapping)
        self.preprocessor = DataPreprocessor(target_variable)
        self.trainer = ModelTrainer(target_variable)
        self.optimizer = HyperparameterOptimizer(target_variable)
    
    def load_data(self, df: pd.DataFrame) -> 'MLPipeline':
        """Загрузка данных"""
        self.df = df.copy()
        return self
    
    def run_eda(self, output_dir: Optional[str] = None) -> 'MLPipeline':
        """Запуск полного EDA"""
        if self.df is None:
            raise ValueError("Данные не загружены. Вызовите load_data() сначала.")
        
        self.analyzer.load_data(self.df).run_full_eda(output_dir)
        return self
    
    def preprocess_data(self, 
                       high_missing_threshold: float = 70.0,
                       multicollinearity_threshold: float = 0.85,
                       max_features: int = 50,
                       test_size: float = 0.2,
                       val_size: float = 0.25,
                       random_state: int = 42,
                       output_dir: Optional[str] = None) -> 'MLPipeline':
        """Полная предобработка данных"""
        if self.df is None:
            raise ValueError("Данные не загружены. Вызовите load_data() сначала.")
        
        (self.preprocessor.load_data(self.df)
         .prepare_target()
         .handle_missing_values(high_missing_threshold)
         .encode_categorical_features()
         .remove_leaky_features()
         .handle_multicollinearity(multicollinearity_threshold)
         .select_features(max_features)
         .clean_column_names()
         .split_data(test_size, val_size, random_state))
        
        (self.X_train, self.y_train,
         self.X_val, self.y_val,
         self.X_test, self.y_test) = self.preprocessor.get_processed_data()
        self.scale_pos_weight = self.preprocessor.get_scale_pos_weight()
        
        if output_dir:
            self.preprocessor.save_preprocessed_data(output_dir)
        
        return self
    
    def train_models(self, optimize_threshold: bool = True) -> 'MLPipeline':
        """Обучение baseline моделей"""
        if self.X_train is None:
            raise ValueError("Данные не предобработаны. Вызовите preprocess_data() сначала.")
        
        self.trainer.train_baseline_models(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            self.scale_pos_weight
        ).compare_models()
        
        if optimize_threshold:
            self.trainer.optimize_threshold()
        
        return self
    
    def optimize_hyperparameters(self, 
                                models: List[str] = ['XGBoost', 'LightGBM', 'CatBoost'],
                                n_iter: int = 25) -> 'MLPipeline':
        """Гиперпараметрическая оптимизация"""
        if self.X_train is None:
            raise ValueError("Данные не предобработаны. Вызовите preprocess_data() сначала.")
        
        display(Markdown("## 🚀 Гиперпараметрическая оптимизация"))
        
        if 'XGBoost' in models:
            self.optimizer.optimize_xgboost(
                self.X_train, self.y_train, self.X_val, self.y_val,
                self.scale_pos_weight, n_iter=n_iter
            )
        
        if 'LightGBM' in models:
            self.optimizer.optimize_lightgbm(
                self.X_train, self.y_train, self.X_val, self.y_val,
                self.scale_pos_weight, n_iter=n_iter
            )
        
        if 'CatBoost' in models:
            self.optimizer.optimize_catboost(
                self.X_train, self.y_train, self.X_val, self.y_val,
                self.scale_pos_weight, n_iter=min(n_iter, 15)  # CatBoost медленнее
            )
        
        return self
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Получить лучшую модель"""
        return self.trainer.get_best_model()
    
    def save_pipeline_state(self, output_dir: str) -> 'MLPipeline':
        """Сохранение состояния пайплайна"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.X_train is not None:
            self.X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
            self.y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
            self.X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
            self.y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
            self.X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
            self.y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        pd.DataFrame([{'scale_pos_weight': self.scale_pos_weight}]).to_json(
            os.path.join(output_dir, 'pipeline_info.json'), orient='records')
        
        display(Markdown(f"✅ Состояние пайплайна сохранено в: {output_dir}"))
        return self
    
    def run_full_pipeline(self, 
                         eda_output_dir: Optional[str] = None,
                         preprocess_output_dir: Optional[str] = None,
                         pipeline_output_dir: Optional[str] = None,
                         optimize_hp: bool = False) -> 'MLPipeline':
        """Запуск полного пайплайна"""
        display(Markdown("# 🚀 Запуск полного ML пайплайна"))
        
        self.run_eda(eda_output_dir)
        
        self.preprocess_data(output_dir=preprocess_output_dir)
        
        self.train_models()
        
        if optimize_hp:
            self.optimize_hyperparameters()
        
        if pipeline_output_dir:
            self.save_pipeline_state(pipeline_output_dir)
        
        display(Markdown("## ✅ Пайплайн успешно завершён!"))
        return self


def quick_eda(df: pd.DataFrame, target_variable: str = 'Смерть') -> DataAnalyzer:
    """Быстрый EDA без сохранения"""
    analyzer = DataAnalyzer(target_variable=target_variable)
    return analyzer.load_data(df).run_full_eda()

def quick_preprocess(df: pd.DataFrame, target_variable: str = 'Смерть', 
                    max_features: int = 50) -> Tuple[pd.DataFrame, pd.Series, 
                                                   pd.DataFrame, pd.Series,
                                                   pd.DataFrame, pd.Series, float]:
    """Быстрая предобработка данных"""
    preprocessor = DataPreprocessor(target_variable=target_variable)
    (preprocessor.load_data(df)
     .prepare_target()
     .handle_missing_values()
     .encode_categorical_features()
     .remove_leaky_features()
     .handle_multicollinearity()
     .select_features(max_features)
     .clean_column_names()
     .split_data())
    
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_processed_data()
    scale_pos_weight = preprocessor.get_scale_pos_weight()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight

def quick_train(X_train: pd.DataFrame, y_train: pd.Series,
               X_val: pd.DataFrame, y_val: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series,
               scale_pos_weight: float) -> Tuple[Dict, Dict]:
    """Быстрое обучение моделей"""
    trainer = ModelTrainer()
    trainer.train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight)
    comparison_df = trainer.compare_models()
    return trainer.models, trainer.results