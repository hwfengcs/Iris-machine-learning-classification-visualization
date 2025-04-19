#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度学习模型模块
包含神经网络模型的定义、训练和评估
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns

class DeepLearningModel:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        tf.random.set_seed(random_seed)
        self.model = None
        self.history = None
    
    def create_mlp_model(self, input_shape):
        """创建多层感知机模型"""
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001),
                        input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, epochs=100, batch_size=16, validation_split=0.2):
        """训练模型"""
        # 将标签进行独热编码
        y_train_categorical = to_categorical(y_train)
        
        # 设置早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_test_categorical = to_categorical(y_test)
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"\n测试集准确率: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    
    def predict(self, X):
        """模型预测"""
        return self.model.predict(X)
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/mlp_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, X_test, y_test, target_names):
        """绘制混淆矩阵"""
        # 获取预测结果
        y_pred = np.argmax(self.predict(X_test), axis=1)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix - MLP Model', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('plots/mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, X_test, y_test, target_names):
        """绘制ROC曲线"""
        y_pred_proba = self.predict(X_test)
        
        plt.figure(figsize=(15, 5))
        plt.suptitle('ROC Curves for MLP Model (One-vs-Rest)', fontsize=16)
        
        for i, class_name in enumerate(target_names):
            plt.subplot(1, 3, i+1)
            
            # 计算当前类别的ROC曲线
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr,
                     label=f'ROC curve (AUC = {roc_auc:.2f})',
                     linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve for {class_name}', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/mlp_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()