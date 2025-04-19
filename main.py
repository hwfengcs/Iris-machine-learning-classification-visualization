#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
鸢尾花分类综合分析主程序
整合数据处理、可视化、机器学习和深度学习模块
"""

import warnings
from data_processor import IrisDataProcessor
from visualizer import IrisVisualizer
from ml_models import MLModels
from dl_models import DeepLearningModel

# 忽略警告信息
warnings.filterwarnings('ignore')

def main():
    # 初始化数据处理器
    data_processor = IrisDataProcessor()
    
    # 加载数据
    X, y, df, feature_names, target_names = data_processor.load_data()
    
    # 数据预处理
    X_train_scaled, X_test_scaled, y_train, y_test = data_processor.preprocess_data(X, y)
    
    # 打印数据信息
    data_processor.print_data_info(X, y, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 数据可视化
    print("\n开始数据可视化...")
    visualizer = IrisVisualizer()
    visualizer.plot_feature_distributions(X, y, feature_names, target_names)
    visualizer.plot_pairplot(df)
    visualizer.plot_correlation_heatmap(df)
    visualizer.plot_boxplots(df, feature_names)
    visualizer.plot_violinplots(df, feature_names)
    visualizer.plot_3d_scatter(df, feature_names)
    visualizer.plot_dimensionality_reduction(X, y)
    
    # 机器学习模型训练和评估
    print("\n开始训练和评估机器学习模型...")
    ml_models = MLModels()
    cv_results = ml_models.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 绘制机器学习模型的评估图
    ml_models.plot_model_comparison()
    ml_models.plot_confusion_matrices(y_test, target_names)
    ml_models.plot_roc_curves(y_test, target_names)
    ml_models.plot_precision_recall_curves(y_test, target_names)
    ml_models.plot_feature_importance(feature_names)
    
    # 深度学习模型训练和评估
    print("\n开始训练和评估深度学习模型...")
    dl_model = DeepLearningModel()
    model = dl_model.create_mlp_model((X_train_scaled.shape[1],))
    dl_model.train(X_train_scaled, y_train)
    
    # 评估深度学习模型
    test_loss, test_accuracy = dl_model.evaluate(X_test_scaled, y_test)
    
    # 绘制深度学习模型的评估图
    dl_model.plot_training_history()
    dl_model.plot_confusion_matrix(X_test_scaled, y_test, target_names)
    dl_model.plot_roc_curves(X_test_scaled, y_test, target_names)

if __name__ == "__main__":
    main()