#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理模块
包含数据加载、预处理等功能
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class IrisDataProcessor:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def load_data(self):
        """加载鸢尾花数据集"""
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # 创建数据框
        df = pd.DataFrame(data=X, columns=feature_names)
        df['species'] = pd.Categorical.from_codes(y, target_names)
        
        return X, y, df, feature_names, target_names
    
    def preprocess_data(self, X, y, test_size=0.2):
        """数据预处理：划分数据集并进行标准化"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def print_data_info(self, X, y, X_train, X_test, y_train, y_test):
        """打印数据集信息"""
        print("数据集基本信息:")
        print(f"样本数量: {X.shape[0]}")
        print(f"特征数量: {X.shape[1]}")
        print(f"目标类别: {', '.join(datasets.load_iris().target_names)}")
        
        print(f"\n训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        print(f"各类别在训练集中的分布: {np.bincount(y_train)}")
        print(f"各类别在测试集中的分布: {np.bincount(y_test)}")