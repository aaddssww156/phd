import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, balanced_accuracy_score)
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class FeatureSelectionAnalyzer:
    def __init__(self, X, y, test_size=0.2, random_state=42, categorial_features=None):
        """
        Инициализация анализатора отбора признаков
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Матрица признаков
        y : pandas.Series или numpy.array
            Целевая переменная (бинарная классификация)
        test_size : float, default=0.2
            Размер тестовой выборки
        random_state : int, default=42
            Фиксатор случайности
        categorial_features : str, default=None
            Список наименований категориальных колонок
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.categorial_features = categorial_features or []
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.forward_results = {}
        self.backward_results = {}
        self.best_threshold = 0.5
        
    def _evaluate_model(self, model, features, method_name, step):
        """
        Оценка модели и сохранение результатов
        """
        start_time = time.time()

        current_cat_features = [f for f in self.categorial_features if f in features]
        
        X_train_subset = self.X_train[features]
        X_test_subset = self.X_test[features]

        model_type = type(model).__name__
        fit_params = {}

        if model_type in ['CatBoostClassifier', 'CatBoostRegressor']:
            fit_params['cat_features'] = current_cat_features
        elif model_type in ['LGBMClassifier', 'LGBMRegressor']:
            for col in current_cat_features:
                if X_train_subset[col].dtype.name != 'category':
                    X_train_subset[col] = X_train_subset[col].astype('category').cat.codes
                    X_test_subset[col] = X_test_subset[col].astype('category').cat.codes
                else:
                    X_train_subset[col] = X_train_subset[col].cat.codes
                    X_test_subset[col] = X_test_subset[col].cat.codes
            fit_params['categorical_feature'] = current_cat_features
        elif model_type in ['XGBCLassifier', 'XGBRegressor']:
            import xgboost
            if xgboost.__version__ < '1.6.0':
                raise ValueError(f"XGBoost >= 1.6.0 required, got {xgboost.__version__}")

            for col in current_cat_features:
                if X_train_subset[col].dtype.name != 'category':
                    X_train_subset[col] = X_train_subset[col].astype('category')
                    X_test_subset[col] = X_test_subset[col].astype('category')
            fit_params['enable_categorial'] = True

        from sklearn.base import clone 
        model_step = clone(model)

        try:
            model_step.fit(X_train_subset, self.y_train, **fit_params)
        except Exception as e:
            raise ValueError(
                f"Ошибка обучения на шаге {step} ({method_name}) с признаками {features}:\n"
                f"Категориальные признаки: {current_cat_features}\n"
                f"Тип модели: {model_type}\n"
                f"Ошибка: {str(e)}"
            ) from e
        
        y_pred_proba = model_step.predict_proba(X_test_subset)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'training_time': time.time() - start_time
        }
        
        feature_importance = {}
        if hasattr(model_step, 'feature_importances_'):
            feature_importance = dict(zip(features, model_step.feature_importances_))
        elif hasattr(model_step, 'coef_'):
            feature_importance = dict(zip(features, np.abs(model_step.coef_[0])))
        
        result = {
            'step': step,
            'features': features.copy(),
            'n_features': len(features),
            'metrics': metrics,
            'feature_importance': feature_importance,
            'y_pred_proba': y_pred_proba.copy(),
            'y_pred': y_pred.copy(),
            'y_true': self.y_test.copy(),
            'model': model_step.__class__.__name__
        }
        
        return result, y_pred_proba
    
    def forward_selection(self, model, max_features=None, cv=3, metric='f1'):
        """
        Forward Selection с сохранением всех результатов
        
        Parameters:
        -----------
        model : sklearn estimator
            Модель для обучения
        max_features : int, optional
            Максимальное количество признаков для отбора
        cv : int, default=3
            Количество фолдов для кросс-валидации (необязательно)
        """
        print("Запуск Forward Selection...")
        
        all_features = list(self.X.columns)
        if max_features is None:
            max_features = len(all_features)
            
        selected_features = []
        remaining_features = all_features.copy()
        best_metric_score = -np.inf
        step = 0
        
        pbar = tqdm(total=min(max_features, len(all_features)))
        
        while len(selected_features) < max_features and remaining_features:
            step += 1
            best_feature = None
            best_score = -np.inf
            
            for feature in remaining_features:
                candidate_features = selected_features + [feature]
                
                result, _ = self._evaluate_model(model, candidate_features, 'forward', step)
                current_metric_score = result['metrics'][metric]
                
                if current_metric_score > best_score:
                    best_score = current_metric_score
                    best_feature = feature
                    best_result = result
            
            if best_score > best_metric_score:
                best_metric_score = best_score
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                
                self.forward_results[len(selected_features)] = best_result
                print(f"Step {step}: Добавлен признак '{best_feature}', {metric}: {best_score:.4f}")
            else:
                print(f"Step {step}: Нет улучшений, останавливаемся")
                break
            
            pbar.update(1)
            pbar.set_postfix({f'{metric}': f'{best_score:.4f}', 'Features': len(selected_features)})
        
        pbar.close()
        print(f"Forward Selection завершен. Выбрано признаков: {len(selected_features)}")
        return selected_features
    
    def backward_elimination(self, model, min_features=1, metric='f1'):
        """
        Backward Elimination с сохранением всех результатов
        
        Parameters:
        -----------
        model : sklearn estimator
            Модель для обучения
        min_features : int, default=1
            Минимальное количество признаков
        """
        print("Запуск Backward Elimination...")
        
        selected_features = list(self.X.columns)
        best_metric_score = -np.inf
        step = 0
        
        pbar = tqdm(total=len(selected_features) - min_features)
        
        while len(selected_features) > min_features:
            step += 1
            worst_feature = None
            best_score = -np.inf
            
            for feature in selected_features:
                candidate_features = [f for f in selected_features if f != feature]
                
                result, _ = self._evaluate_model(model, candidate_features, 'backward', step)
                current_metric_score = result['metrics'][metric]
                
                if current_metric_score > best_score:
                    best_score = current_metric_score
                    worst_feature = feature
                    best_result = result
            
            if best_score > best_metric_score:
                best_metric_score = best_score
                selected_features.remove(worst_feature)
                
                self.backward_results[len(selected_features)] = best_result
                print(f"Step {step}: Удален признак '{worst_feature}', {metric}: {best_score:.4f}")
            else:
                print(f"Step {step}: Удаление ухудшает результат, останавливаемся")
                break
            
            pbar.update(1)
            pbar.set_postfix({f'{metric}': f'{best_score:.4f}', 'Features': len(selected_features)})
        
        pbar.close()
        print(f"Backward Elimination завершен. Осталось признаков: {len(selected_features)}")
        return selected_features
    
    def find_optimal_threshold(self, method='forward', n_features=None):
        """
        Поиск оптимального порога для бинарной классификации
        
        Parameters:
        -----------
        method : str, 'forward' or 'backward'
            Метод отбора признаков
        n_features : int, optional
            Количество признаков для анализа
        """
        if method == 'forward' and n_features is None:
            n_features = max(self.forward_results.keys())
        elif method == 'backward' and n_features is None:
            n_features = min(self.backward_results.keys())
        
        if method == 'forward':
            result = self.forward_results[n_features]
        else:
            result = self.backward_results[n_features]
        
        y_true = result['y_true']
        y_pred_proba = result['y_pred_proba']
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = -1
        best_threshold = 0.5
        # best_balanced_accuracy = -1
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            # balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            if f1 > best_f1:
            # if balanced_accuracy > best_balanced_accuracy:
                # best_balanced_accuracy = balanced_accuracy
                best_f1 = f1
                best_threshold = threshold
        
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        best_j_threshold = thresholds_roc[np.argmax(j_scores)]
        
        self.best_threshold = best_threshold
        
        print(f"=== Оптимальные пороги ({method}, {n_features} признаков) ===")
        print(f"Лучший Threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
        print(f"По Youden's J: {best_j_threshold:.4f}")
        
        return best_threshold, best_j_threshold
    
    def visualize_probability_distribution(self, method='forward', n_features=None):
        """
        Визуализация распределения вероятностей
        
        Parameters:
        -----------
        method : str, 'forward' or 'backward'
            Метод отбора признаков
        n_features : int, optional
            Количество признаков для визуализации
        """
        if method == 'forward' and n_features is None:
            n_features = max(self.forward_results.keys())
        elif method == 'backward' and n_features is None:
            n_features = min(self.backward_results.keys())
        
        if method == 'forward':
            result = self.forward_results[n_features]
        else:
            result = self.backward_results[n_features]
        
        y_true = result['y_true']
        y_pred_proba = result['y_pred_proba']
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.histplot(data=pd.DataFrame({
            'Probability': y_pred_proba,
            'Class': ['Positive' if x == 1 else 'Negative' for x in y_true]
        }), x='Probability', hue='Class', bins=30, kde=True, alpha=0.6)
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Default threshold (0.5)')
        if hasattr(self, 'best_threshold'):
            plt.axvline(x=self.best_threshold, color='g', linestyle='-', alpha=0.7, 
                       label=f'Optimal threshold ({self.best_threshold:.3f})')
        plt.title(f'Распределение вероятностей ({method}, {n_features} признаков)')
        plt.xlabel('Вероятность класса 1')
        plt.ylabel('Частота')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC кривая')
        plt.legend(loc="lower right")
        
        plt.subplot(2, 2, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall кривая')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        y_pred_optimal = (y_pred_proba >= self.best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Матрица ошибок (threshold={self.best_threshold:.3f})')
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        df_probs = pd.DataFrame({
            'Probability': y_pred_proba,
            'Class': y_true
        })
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='Class', y='Probability', data=df_probs)
        plt.title('Boxplot вероятностей по классам')
        plt.xlabel('Класс')
        plt.ylabel('Вероятность')
        
        plt.subplot(1, 2, 2)

        quantiles = np.percentile(y_pred_proba, [0, 25, 50, 75, 100])
        plt.hist(y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
        for q in quantiles:
            plt.axvline(x=q, color='r', linestyle='--', alpha=0.5)
        plt.title('Гистограмма вероятностей с квантилями')
        plt.xlabel('Вероятность')
        plt.ylabel('Частота')
        
        plt.tight_layout()
        plt.show()
    
    def compare_feature_selection_methods(self, metric='f1'):
        """
        Сравнение результатов forward selection и backward elimination
        """
        if not self.forward_results and not self.backward_results:
            print("Нет результатов для сравнения. Запустите сначала forward_selection и backward_elimination.")
            return
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        
        if self.forward_results:
            forward_x = sorted(self.forward_results.keys())
            forward_y = [self.forward_results[x]['metrics']['roc_auc'] for x in forward_x]
            plt.plot(forward_x, forward_y, 'o-', label='Forward Selection', color='blue')
        
        if self.backward_results:
            backward_x = sorted(self.backward_results.keys())
            backward_y = [self.backward_results[x]['metrics']['roc_auc'] for x in backward_x]
            plt.plot(backward_x, backward_y, 's-', label='Backward Elimination', color='red')
        
        plt.xlabel('Количество признаков')
        plt.ylabel('ROC-AUC')
        plt.title('Сравнение ROC-AUC по количеству признаков')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        
        if self.forward_results:
            forward_f1 = [self.forward_results[x]['metrics']['f1'] for x in forward_x]
            plt.plot(forward_x, forward_f1, 'o-', label='Forward Selection', color='blue')
        
        if self.backward_results:
            backward_f1 = [self.backward_results[x]['metrics']['f1'] for x in backward_x]
            plt.plot(backward_x, backward_f1, 's-', label='Backward Elimination', color='red')
        
        plt.xlabel('Количество признаков')
        plt.ylabel('F1 Score')
        plt.title('Сравнение F1 Score по количеству признаков')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        
        if self.forward_results:
            forward_balanced_accuracy = [self.forward_results[x]['metrics']['balanced_accuracy'] for x in forward_x]
            plt.plot(forward_x, forward_balanced_accuracy, 'o-', label='Forward Selection', color='blue')
        
        if self.backward_results:
            backward_balanced_accuracy = [self.backward_results[x]['metrics']['balanced_accuracy'] for x in backward_x]
            plt.plot(backward_x, backward_balanced_accuracy, 's-', label='Backward Elimination', color='red')
        
        plt.xlabel('Количество признаков')
        plt.ylabel('Balanced Accuracy')
        plt.title('Сравнение Balanced Accuracy по количеству признаков')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        
        if self.forward_results:
            forward_time = [self.forward_results[x]['metrics']['training_time'] for x in forward_x]
            plt.plot(forward_x, forward_time, 'o-', label='Forward Selection', color='blue')
        
        if self.backward_results:
            backward_time = [self.backward_results[x]['metrics']['training_time'] for x in backward_x]
            plt.plot(backward_x, backward_time, 's-', label='Backward Elimination', color='red')
        
        plt.xlabel('Количество признаков')
        plt.ylabel('Время обучения (сек)')
        plt.title('Сравнение времени обучения')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        
        best_forward = max(self.forward_results.items(), key=lambda x: x[1]['metrics'][metric])[1]
        best_backward = max(self.backward_results.items(), key=lambda x: x[1]['metrics'][metric])[1]
        
        all_features = set(list(best_forward['feature_importance'].keys()) + 
                          list(best_backward['feature_importance'].keys()))
        
        forward_importance = [best_forward['feature_importance'].get(f, 0) for f in all_features]
        backward_importance = [best_backward['feature_importance'].get(f, 0) for f in all_features]
        
        x = np.arange(len(all_features))
        width = 0.35
        
        plt.bar(x - width/2, forward_importance, width, label='Forward Selection')
        plt.bar(x + width/2, backward_importance, width, label='Backward Elimination')
        plt.xlabel('Признаки')
        plt.ylabel('Важность')
        plt.title('Сравнение важности признаков')
        plt.xticks(x, list(all_features), rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_best_features(self, method='forward', metric='f1'):
        """
        Получение лучших признаков по заданной метрике
        
        Parameters:
        -----------
        method : str, 'forward' or 'backward'
            Метод отбора признаков
        metric : str, default='roc_auc'
            Метрика для выбора лучших признаков
        """
        if method == 'forward':
            results = self.forward_results
        else:
            results = self.backward_results
        
        if not results:
            print(f"Нет результатов для метода {method}. Запустите сначала отбор признаков.")
            return None
        
        best_result = max(results.items(), key=lambda x: x[1]['metrics'][metric])[1]
        
        print(f"=== Лучшие признаки ({method}) ===")
        print(f"Количество признаков: {best_result['n_features']}")
        print(f"Лучшая {metric}: {best_result['metrics'][metric]:.4f}")
        print(f"Признаки: {best_result['features']}")
        
        if best_result['feature_importance']:
            print("\nВажность признаков:")
            sorted_importance = sorted(best_result['feature_importance'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance:
                print(f"  {feature}: {importance:.4f}")
        
        return best_result['features'], best_result
    
    def save_results_to_excel(self, filename='feature_selection_results.xlsx'):
        """
        Сохранение всех результатов в Excel файл
        
        Parameters:
        -----------
        filename : str, default='feature_selection_results.xlsx'
            Имя файла для сохранения
        """
        with pd.ExcelWriter(filename) as writer:
            if self.forward_results:
                forward_df = []
                for n_features, result in self.forward_results.items():
                    row = {
                        'Method': 'Forward',
                        'Step': result['step'],
                        'N_Features': result['n_features'],
                        'Features': ', '.join(result['features']),
                        'Accuracy': result['metrics']['accuracy'],
                        'Precision': result['metrics']['precision'],
                        'Recall': result['metrics']['recall'],
                        'F1': result['metrics']['f1'],
                        'ROC_AUC': result['metrics']['roc_auc'],
                        'Training_Time': result['metrics']['training_time']
                    }
                    
                    for feature, importance in result['feature_importance'].items():
                        row[f'Importance_{feature}'] = importance
                    
                    forward_df.append(row)
                
                pd.DataFrame(forward_df).to_excel(writer, sheet_name='Forward_Selection', index=False)
            
            if self.backward_results:
                backward_df = []
                for n_features, result in self.backward_results.items():
                    row = {
                        'Method': 'Backward',
                        'Step': result['step'],
                        'N_Features': result['n_features'],
                        'Features': ', '.join(result['features']),
                        'Accuracy': result['metrics']['accuracy'],
                        'Precision': result['metrics']['precision'],
                        'Recall': result['metrics']['recall'],
                        'F1': result['metrics']['f1'],
                        'ROC_AUC': result['metrics']['roc_auc'],
                        'Training_Time': result['metrics']['training_time']
                    }
                    
                    for feature, importance in result['feature_importance'].items():
                        row[f'Importance_{feature}'] = importance
                    
                    backward_df.append(row)
                
                pd.DataFrame(backward_df).to_excel(writer, sheet_name='Backward_Elimination', index=False)
            
            if self.forward_results and self.backward_results:
                comparison_data = []
                
                best_forward = max(self.forward_results.items(), key=lambda x: x[1]['metrics']['f1'])[1]
                best_backward = max(self.backward_results.items(), key=lambda x: x[1]['metrics']['f1'])[1]
                
                comparison_data.append({
                    'Method': 'Forward_Selection',
                    'N_Features': best_forward['n_features'],
                    'Features': ', '.join(best_forward['features']),
                    'ROC_AUC': best_forward['metrics']['roc_auc'],
                    'F1': best_forward['metrics']['f1'],
                    'Accuracy': best_forward['metrics']['accuracy']
                })
                
                comparison_data.append({
                    'Method': 'Backward_Elimination',
                    'N_Features': best_backward['n_features'],
                    'Features': ', '.join(best_backward['features']),
                    'ROC_AUC': best_backward['metrics']['roc_auc'],
                    'F1': best_backward['metrics']['f1'],
                    'Accuracy': best_backward['metrics']['accuracy']
                })
                
                pd.DataFrame(comparison_data).to_excel(writer, sheet_name='Best_Results_Comparison', index=False)
        
        print(f"Результаты успешно сохранены в файл: {filename}")


# Пример использования:

# Загрузка данных (пример)
# from sklearn.datasets import make_classification

# # Генерация примера данных
# X, y = make_classification(
#     n_samples=1000,
#     n_features=20,
#     n_informative=10,
#     n_redundant=5,
#     n_classes=2,
#     random_state=42
# )

# feature_names = [f'feature_{i}' for i in range(20)]
# X_df = pd.DataFrame(X, columns=feature_names)
# y_series = pd.Series(y)

# # Инициализация анализатора
# analyzer = FeatureSelectionAnalyzer(X_df, y_series, test_size=0.2, random_state=42)

# # Выбор модели
# model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# # Запуск forward selection
# print("=== FORWARD SELECTION ===")
# forward_features = analyzer.forward_selection(model_rf, max_features=15)

# # Запуск backward elimination
# print("\n=== BACKWARD ELIMINATION ===")
# backward_features = analyzer.backward_elimination(model_rf, min_features=5)

# # Сравнение методов
# print("\n=== СРАВНЕНИЕ МЕТОДОВ ===")
# analyzer.compare_feature_selection_methods()

# # Поиск оптимального порога для лучшей модели forward selection
# print("\n=== ПОИСК ОПТИМАЛЬНОГО ПОРОГА ===")
# best_threshold, _ = analyzer.find_optimal_threshold(method='forward')

# # Визуализация распределения вероятностей
# print("\n=== ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ВЕРОЯТНОСТЕЙ ===")
# analyzer.visualize_probability_distribution(method='forward')

# # Получение лучших признаков
# print("\n=== ЛУЧШИЕ ПРИЗНАКИ ===")
# best_forward_features, best_forward_result = analyzer.get_best_features(method='forward')
# best_backward_features, best_backward_result = analyzer.get_best_features(method='backward')

# # Сохранение результатов
# print("\n=== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")
# analyzer.save_results_to_excel('feature_selection_analysis.xlsx')

# print("\n=== ЗАКЛЮЧЕНИЕ ===")
# print(f"Лучший ROC AUC (Forward): {best_forward_result['metrics']['roc_auc']:.4f} с {len(best_forward_features)} признаками")
# print(f"Лучший ROC AUC (Backward): {best_backward_result['metrics']['roc_auc']:.4f} с {len(best_backward_features)} признаками")
# print(f"Оптимальный порог для бинарной классификации: {best_threshold:.4f}")