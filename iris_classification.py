#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
鸢尾花分类综合分析
包含多种机器学习和深度学习方法，以及高质量可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.base import clone
# UMAP相关代码已移除
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scikitplot as skplt
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 设置显示风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 创建结果目录
if not os.path.exists('plots'):
    os.makedirs('plots')
    
# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 创建数据框
df = pd.DataFrame(data=X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# 打印数据集基本信息
print("数据集基本信息:")
print(f"样本数量: {X.shape[0]}")
print(f"特征数量: {X.shape[1]}")
print(f"目标类别: {', '.join(target_names)}")
print("\n数据前5行:")
print(df.head())

# 数据预处理
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"各类别在训练集中的分布: {np.bincount(y_train)}")
print(f"各类别在测试集中的分布: {np.bincount(y_test)}")

# ==================== 数据可视化 ====================
print("\n开始进行数据可视化...")

# 特征分布可视化
plt.figure(figsize=(12, 10))
plt.suptitle('Feature Distributions by Species', fontsize=16)

for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    for species in range(3):
        plt.hist(X[y == species, i], alpha=0.7, label=target_names[species], 
                 bins=15, edgecolor='black', linewidth=0.5)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/feature_distributions.png', dpi=300, bbox_inches='tight')

# 散点矩阵图
plt.figure(figsize=(12, 10))
sns.set(style="whitegrid", font_scale=1.2)
scatter_plot = sns.pairplot(df, hue='species', palette='viridis', 
                            height=2.5, markers=["o", "s", "D"],
                            plot_kws={'s': 80, 'edgecolor': 'k', 'linewidth': 0.5, 'alpha': 0.7})
scatter_plot.fig.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=16)
plt.savefig('plots/pairwise_scatter.png', dpi=300, bbox_inches='tight')

# 相关性热图
plt.figure(figsize=(10, 8))
correlation = df.drop('species', axis=1).corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
heatmap = sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                      annot=True, fmt=".2f", square=True, linewidths=.5)
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')

# 箱线图
plt.figure(figsize=(14, 10))
plt.suptitle('Feature Distribution Boxplots by Species', fontsize=16)

for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=df, palette='viridis',
                flierprops={'marker': 'o', 'markersize': 8})
    sns.stripplot(x='species', y=feature, data=df, size=5, color=".3", alpha=0.5)
    plt.title(f'{feature}', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/feature_boxplots.png', dpi=300, bbox_inches='tight')

# 特征分布小提琴图
plt.figure(figsize=(14, 10))
plt.suptitle('Feature Distribution Violin Plots by Species', fontsize=16)

for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=df, palette='viridis', 
                  inner='quart', linewidth=1.5)
    plt.title(f'{feature}', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/feature_violinplots.png', dpi=300, bbox_inches='tight')

# 使用Plotly创建3D散点图
fig = px.scatter_3d(df, x=feature_names[0], y=feature_names[1], z=feature_names[2],
                   color='species', symbol='species', opacity=0.7,
                   title='3D Scatter Plot of Iris Features',
                   labels={feature_names[0]: feature_names[0],
                          feature_names[1]: feature_names[1],
                          feature_names[2]: feature_names[2]},
                   color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(legend_title_text='Species')
fig.write_html('plots/3d_scatter.html')

# 降维可视化
print("\n进行降维分析...")

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                     edgecolor='k', s=150, alpha=0.7)
plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', fontsize=12)
plt.title('PCA of Iris Dataset', fontsize=14)
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
X_tsne = tsne.fit_transform(X)

plt.subplot(2, 2, 2)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                     edgecolor='k', s=150, alpha=0.7)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.title('t-SNE of Iris Dataset', fontsize=14)
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)

# UMAP降维部分已移除
plt.subplot(2, 2, 3)
plt.text(0.5, 0.5, '此部分已移除', 
         horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=14)
plt.axis('off')

# PCA与原始特征的关系
plt.subplot(2, 2, 4)
components = pca.components_
features = feature_names
plt.imshow(components, cmap='viridis')
plt.yticks([0, 1], [f'PC1 ({explained_variance[0]:.2%})', f'PC2 ({explained_variance[1]:.2%})'])
plt.xticks(range(len(features)), features, rotation=45, ha='right')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
plt.title('PCA Components Heatmap', fontsize=14)
plt.colorbar(label='Component Coefficient')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Dimensionality Reduction Techniques Comparison', fontsize=16, y=0.98)
plt.savefig('plots/dimensionality_reduction.png', dpi=300, bbox_inches='tight')

# ==================== 机器学习模型 ====================
print("\n开始训练和评估机器学习模型...")

# 创建模型字典
models = {
    "Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "SVM": SVC(probability=True, random_state=RANDOM_SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED)
}

# 模型评估结果
model_accuracies = {}
model_test_accuracies = {}
all_predictions = {}
all_probabilities = {}

# 用于创建决策边界可视化的网格
h = 0.02  # 步长
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh_points = np.c_[xx.ravel(), yy.ravel()]

# 创建一个图片来显示所有模型的决策边界
plt.figure(figsize=(20, 15))
plt.suptitle('Decision Boundaries of Various Machine Learning Models', fontsize=20, y=0.98)

# K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# 收集每个模型的交叉验证得分
cv_results = {}

# 对每个模型进行训练和评估
for i, (name, model) in enumerate(models.items()):
    print(f"\n训练 {name} 模型...")
    
    # 使用交叉验证获取平均性能
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    cv_results[name] = cv_scores
    model_accuracies[name] = cv_scores.mean()
    
    # 在全部训练数据上训练模型
    model.fit(X_train_scaled, y_train)
    
    # 在测试集上评估
    y_pred = model.predict(X_test_scaled)
    all_predictions[name] = y_pred
    model_test_accuracies[name] = accuracy_score(y_test, y_pred)
    
    # 对于支持概率输出的模型，收集预测概率
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
        all_probabilities[name] = y_proba
    
    # 创建决策边界可视化
    plt.subplot(3, 3, i + 1)
    
    # 为了可视化，我们使用PCA降维后的数据
    # 训练一个新模型在PCA降维后的数据上
    pca_model = clone(model)
    pca_model.fit(X_pca, y)
    
    # 预测网格点的类别
    if hasattr(pca_model, "decision_function"):
        Z = pca_model.decision_function(mesh_points)
    else:
        Z = pca_model.predict(mesh_points)
    
    # 将结果整形为网格的形状
    if Z.ndim > 1 and Z.shape[1] > 1:  # 多类别情况
        Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和样本点
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='viridis', s=80, alpha=0.9)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(f'Principal Component 1', fontsize=12)
    plt.ylabel(f'Principal Component 2', fontsize=12)
    plt.title(f'{name} (Test Acc: {model_test_accuracies[name]:.2%})', fontsize=14)
    plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/model_decision_boundaries.png', dpi=300, bbox_inches='tight')

# 创建一个图表来比较所有模型的准确率
plt.figure(figsize=(12, 8))
model_names = list(model_accuracies.keys())
cv_mean_accuracies = [model_accuracies[name] for name in model_names]
test_accuracies = [model_test_accuracies[name] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, cv_mean_accuracies, width, label='Cross-Validation Accuracy', color='royalblue', edgecolor='black', linewidth=1.5, alpha=0.8)
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='lightcoral', edgecolor='black', linewidth=1.5, alpha=0.8)

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

# 创建箱线图比较交叉验证结果
plt.figure(figsize=(12, 8))
cv_data = []
for name in model_names:
    for score in cv_results[name]:
        cv_data.append({'Model': name, 'Accuracy': score})
cv_df = pd.DataFrame(cv_data)

sns.boxplot(x='Model', y='Accuracy', data=cv_df, palette='viridis')
plt.title('Cross-Validation Accuracy Distribution by Model', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cv_accuracy_distribution.png', dpi=300, bbox_inches='tight')

# 为每个模型创建混淆矩阵
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Confusion Matrices for Different Models', fontsize=20, y=0.98)
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    if i < len(axes):
        y_pred = all_predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=target_names, yticklabels=target_names)
        axes[i].set_title(f'{name} (Accuracy: {model_test_accuracies[name]:.2%})', fontsize=14)
        axes[i].set_xlabel('Predicted Label', fontsize=12)
        axes[i].set_ylabel('True Label', fontsize=12)

# 隐藏空余的子图
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')

# 创建ROC曲线
plt.figure(figsize=(15, 10))
plt.suptitle('ROC Curves for Different Models (One-vs-Rest)', fontsize=18)

for i, class_id in enumerate(range(3)):
    plt.subplot(1, 3, i+1)
    
    for name, model in models.items():
        if name in all_probabilities:
            y_prob = all_probabilities[name][:, class_id]
            
            # 计算ROC曲线
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

# 创建精确率-召回率曲线
plt.figure(figsize=(15, 10))
plt.suptitle('Precision-Recall Curves for Different Models (One-vs-Rest)', fontsize=18)

for i, class_id in enumerate(range(3)):
    plt.subplot(1, 3, i+1)
    
    for name, model in models.items():
        if name in all_probabilities:
            y_prob = all_probabilities[name][:, class_id]
            
            # 计算精确率-召回率曲线
            precision, recall, _ = precision_recall_curve(y_test == class_id, y_prob)
            average_precision = average_precision_score(y_test == class_id, y_prob)
            
            plt.plot(recall, precision, lw=2, label=f'{name} (AP = {average_precision:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve for {target_names[class_id]}', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/precision_recall_curves.png', dpi=300, bbox_inches='tight')

# 对于决策树和随机森林，可视化特征重要性
if "Random Forest" in models:
    plt.figure(figsize=(10, 6))
    
    # 获取特征重要性
    importances = models["Random Forest"].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.title('Feature Importance in Random Forest Model', fontsize=16)
    plt.bar(range(X.shape[1]), importances[indices], align='center', alpha=0.7, color='royalblue', edgecolor='black')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/feature_importance_rf.png', dpi=300, bbox_inches='tight')

# SHAP值分析部分已移除

# 创建学习曲线，以评估模型性能随训练集大小的变化
plt.figure(figsize=(15, 10))
plt.suptitle('Learning Curves for Different Models', fontsize=18)

for i, (name, model) in enumerate(list(models.items())[:6]):  # 选择前6个模型来展示
    plt.subplot(2, 3, i+1)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_scaled, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='royalblue', label='Training accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='royalblue')
    
    plt.plot(train_sizes, test_mean, 'o-', color='lightcoral', label='Cross-validation accuracy')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='lightcoral')
    
    plt.title(f'Learning Curve for {name}', fontsize=14)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/learning_curves.png', dpi=300, bbox_inches='tight')

# ==================== 深度学习模型 ====================
print("\n开始训练和评估深度学习模型...")

# 将标签进行独热编码
from tensorflow.keras.utils import to_categorical
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# 设置回调函数
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 创建多层感知机模型
def create_mlp_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train_scaled.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 训练深度学习模型
mlp_model = create_mlp_model()

history = mlp_model.fit(
    X_train_scaled, y_train_categorical,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 评估深度学习模型
y_pred_proba_mlp = mlp_model.predict(X_test_scaled)
y_pred_mlp = np.argmax(y_pred_proba_mlp, axis=1)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"\n深度学习模型准确率: {mlp_accuracy:.4f}")

# 绘制训练历史
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/mlp_training_history.png', dpi=300, bbox_inches='tight')

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - MLP Model', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('plots/mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')

# 为MLP模型创建ROC曲线
plt.figure(figsize=(15, 5))
plt.suptitle('ROC Curves for MLP Model (One-vs-Rest)', fontsize=16)

for i, class_id in enumerate(range(3)):
    plt.subplot(1, 3, i+1)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test == class_id, y_pred_proba_mlp[:, class_id])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve for {target_names[class_id]}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/mlp_roc_curves.png', dpi=300, bbox_inches='tight')

# 尝试不同的神经网络架构
architectures = [
    ('Simple', [16, 8]),
    ('Medium', [32, 16, 8]),
    ('Complex', [64, 32, 16, 8])
]

architecture_histories = {}
architecture_accuracies = {}

for name, hidden_layers in architectures:
    print(f"\n训练 {name} 神经网络架构...")
    
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(X_train_scaled.shape[1],)))
    
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train_scaled, y_train_categorical,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    architecture_histories[name] = history
    
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    architecture_accuracies[name] = accuracy_score(y_test, y_pred)

# 比较不同架构的性能
plt.figure(figsize=(12, 8))

for i, (name, history) in enumerate(architecture_histories.items()):
    plt.subplot(len(architecture_histories), 2, 2*i+1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{name} Architecture - Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(len(architecture_histories), 2, 2*i+2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{name} Architecture - Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/architecture_comparison.png', dpi=300, bbox_inches='tight')

# 比较不同架构的测试准确率
plt.figure(figsize=(10, 6))
names = list(architecture_accuracies.keys())
accuracies = list(architecture_accuracies.values())

plt.bar(names, accuracies, color='lightseagreen', edgecolor='black', alpha=0.7)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)

plt.xlabel('Neural Network Architecture', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.title('Performance Comparison of Different Neural Network Architectures', fontsize=16)
plt.ylim(0.9, 1.05)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/architecture_accuracy_comparison.png', dpi=300, bbox_inches='tight')

# ==================== 集成模型 ====================
print("\n开始训练和评估集成模型...")

# 创建投票分类器
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
        ('svm', SVC(probability=True, random_state=RANDOM_SEED)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED))
    ],
    voting='soft'
)

# 训练集成模型
voting_clf.fit(X_train_scaled, y_train)

# 评估集成模型
y_pred_voting = voting_clf.predict(X_test_scaled)
y_pred_proba_voting = voting_clf.predict_proba(X_test_scaled)
voting_accuracy = accuracy_score(y_test, y_pred_voting)

print(f"集成模型准确率: {voting_accuracy:.4f}")

# 创建混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_voting)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Ensemble Model', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('plots/ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')

# 为集成模型创建ROC曲线
plt.figure(figsize=(15, 5))
plt.suptitle('ROC Curves for Ensemble Model (One-vs-Rest)', fontsize=16)

for i, class_id in enumerate(range(3)):
    plt.subplot(1, 3, i+1)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test == class_id, y_pred_proba_voting[:, class_id])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve for {target_names[class_id]}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/ensemble_roc_curves.png', dpi=300, bbox_inches='tight')

# ==================== 模型对比和总结 ====================
print("\n进行所有模型的比较...")

# 将深度学习和集成模型加入模型对比
all_model_accuracies = model_test_accuracies.copy()
all_model_accuracies['MLP'] = mlp_accuracy
all_model_accuracies['Ensemble'] = voting_accuracy

# 创建最终模型比较图
plt.figure(figsize=(14, 8))
model_names = list(all_model_accuracies.keys())
accuracies = list(all_model_accuracies.values())

# 按准确率从高到低排序
sorted_indices = np.argsort(accuracies)[::-1]
sorted_model_names = [model_names[i] for i in sorted_indices]
sorted_accuracies = [accuracies[i] for i in sorted_indices]

# 使用渐变色来表示不同的准确率
colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_accuracies)))

plt.bar(sorted_model_names, sorted_accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

for i, v in enumerate(sorted_accuracies):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)

plt.xlabel('Models', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.title('Comparative Performance of All Models', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylim(0.9, 1.05)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/final_model_comparison.png', dpi=300, bbox_inches='tight')

# 创建最终的预测结果表格
final_results = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies
})
final_results = final_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\n所有模型准确率排序:")
print(final_results)

# 保存模型比较结果为CSV
final_results.to_csv('model_comparison_results.csv', index=False)

print("\n分析完成! 所有可视化结果已保存到 plots/ 目录")