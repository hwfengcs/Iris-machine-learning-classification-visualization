#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习模型模块
包含所有传统机器学习模型的训练和评估
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

class MLModels:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.models = {
            "Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                                    max_iter=1000, random_state=random_seed),
            "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_seed),
            "SVM": SVC(probability=True, random_state=random_seed),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=random_seed),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=100,
                min_gain_to_split=0.1,
                min_data_in_leaf=5,
                max_depth=5,
                num_leaves=20,
                random_state=random_seed)
        }
        self.model_accuracies = {}
        self.model_test_accuracies = {}
        self.all_predictions = {}
        self.all_probabilities = {}
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """训练和评估所有模型"""
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"\n训练 {name} 模型...")
            
            # 使用交叉验证获取平均性能
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
            cv_results[name] = cv_scores
            self.model_accuracies[name] = cv_scores.mean()
            
            # 在全部训练数据上训练模型
            model.fit(X_train, y_train)
            
            # 在测试集上评估
            y_pred = model.predict(X_test)
            self.all_predictions[name] = y_pred
            self.model_test_accuracies[name] = accuracy_score(y_test, y_pred)
            
            # 对于支持概率输出的模型，收集预测概率
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                self.all_probabilities[name] = y_proba
        
        return cv_results
    
    def plot_model_comparison(self):
        """绘制模型性能比较图"""
        plt.figure(figsize=(12, 8))
        model_names = list(self.model_accuracies.keys())
        cv_mean_accuracies = [self.model_accuracies[name] for name in model_names]
        test_accuracies = [self.model_test_accuracies[name] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, cv_mean_accuracies, width, label='Cross-Validation Accuracy',
                color='royalblue', edgecolor='black', linewidth=1.5, alpha=0.8)
        plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy',
                color='lightcoral', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        for i, v in enumerate(cv_mean_accuracies):
            plt.text(i - width/2, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)
        
        for i, v in enumerate(test_accuracies):
            plt.text(i + width/2, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)
        
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xticks(x, model_names, rotation=45, ha='right', fontsize=12)
        plt.ylim(0.8, 1.05)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('plots/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, y_test, target_names):
        """绘制混淆矩阵"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Confusion Matrices for Different Models', fontsize=20, y=0.98)
        axes = axes.ravel()
        
        for i, (name, model) in enumerate(self.models.items()):
            if i < len(axes):
                y_pred = self.all_predictions[name]
                cm = confusion_matrix(y_test, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                            cbar=False, xticklabels=target_names, yticklabels=target_names)
                axes[i].set_title(f'{name} (Accuracy: {self.model_test_accuracies[name]:.2%})',
                                 fontsize=14)
                axes[i].set_xlabel('Predicted Label', fontsize=12)
                axes[i].set_ylabel('True Label', fontsize=12)
        
        # 隐藏空余的子图
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_test, target_names):
        """绘制ROC曲线"""
        plt.figure(figsize=(15, 10))
        plt.suptitle('ROC Curves for Different Models (One-vs-Rest)', fontsize=18)
        
        for i, class_id in enumerate(range(3)):
            plt.subplot(1, 3, i+1)
            
            for name, model in self.models.items():
                if name in self.all_probabilities:
                    y_prob = self.all_probabilities[name][:, class_id]
                    
                    fpr, tpr, _ = roc_curve(y_test == class_id, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve for {target_names[class_id]}', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, y_test, target_names):
        """绘制精确率-召回率曲线"""
        plt.figure(figsize=(15, 10))
        plt.suptitle('Precision-Recall Curves for Different Models (One-vs-Rest)', fontsize=18)
        
        for i, class_id in enumerate(range(3)):
            plt.subplot(1, 3, i+1)
            
            for name, model in self.models.items():
                if name in self.all_probabilities:
                    y_prob = self.all_probabilities[name][:, class_id]
                    
                    precision, recall, _ = precision_recall_curve(y_test == class_id, y_prob)
                    average_precision = average_precision_score(y_test == class_id, y_prob)
                    
                    plt.plot(recall, precision, lw=2,
                             label=f'{name} (AP = {average_precision:.2f})')
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title(f'Precision-Recall Curve for {target_names[class_id]}', fontsize=14)
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_names):
        """绘制特征重要性（仅限随机森林模型）"""
        if "Random Forest" in self.models:
            plt.figure(figsize=(10, 6))
            
            importances = self.models["Random Forest"].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title('Feature Importance in Random Forest Model', fontsize=16)
            plt.bar(range(len(feature_names)), importances[indices], align='center',
                    alpha=0.7, color='royalblue', edgecolor='black')
            plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices],
                       rotation=45, ha='right')
            plt.xlabel('Features', fontsize=14)
            plt.ylabel('Importance', fontsize=14)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/feature_importance_rf.png', dpi=300, bbox_inches='tight')
            plt.close()