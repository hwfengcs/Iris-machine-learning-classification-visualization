#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块
包含所有数据可视化相关的功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class IrisVisualizer:
    def __init__(self):
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置显示风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        
        # 创建结果目录
        if not os.path.exists('plots'):
            os.makedirs('plots')
    
    def plot_feature_distributions(self, X, y, feature_names, target_names):
        """绘制特征分布图"""
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
        plt.close()
    
    def plot_pairplot(self, df):
        """绘制散点矩阵图"""
        sns.set(style="whitegrid", font_scale=1.2)
        scatter_plot = sns.pairplot(df, hue='species', palette='viridis',
                                   height=2.5, markers=["o", "s", "D"],
                                   plot_kws={'s': 80, 'edgecolor': 'k', 'linewidth': 0.5, 'alpha': 0.7})
        scatter_plot.fig.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=16)
        plt.savefig('plots/pairwise_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, df):
        """绘制相关性热图"""
        plt.figure(figsize=(10, 8))
        correlation = df.drop('species', axis=1).corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt=".2f", square=True, linewidths=.5)
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_boxplots(self, df, feature_names):
        """绘制箱线图"""
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
        plt.close()
    
    def plot_violinplots(self, df, feature_names):
        """绘制小提琴图"""
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
        plt.close()
    
    def plot_3d_scatter(self, df, feature_names):
        """绘制3D散点图"""
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
    
    def plot_dimensionality_reduction(self, X, y, random_seed=42):
        """绘制降维可视化"""
        plt.figure(figsize=(12, 10))
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                             edgecolor='k', s=150, alpha=0.7)
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', fontsize=12)
        plt.title('PCA of Iris Dataset', fontsize=14)
        plt.colorbar(scatter, label='Species')
        plt.grid(True, alpha=0.3)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=random_seed)
        X_tsne = tsne.fit_transform(X)
        
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis',
                             edgecolor='k', s=150, alpha=0.7)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE of Iris Dataset', fontsize=14)
        plt.colorbar(scatter, label='Species')
        plt.grid(True, alpha=0.3)
        
        # PCA与原始特征的关系
        plt.subplot(2, 2, 4)
        components = pca.components_
        plt.imshow(components, cmap='viridis')
        plt.yticks([0, 1], [f'PC1 ({explained_variance[0]:.2%})',
                            f'PC2 ({explained_variance[1]:.2%})'])
        plt.xticks(range(X.shape[1]), range(1, X.shape[1] + 1),
                   rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Principal Components', fontsize=12)
        plt.title('PCA Components Heatmap', fontsize=14)
        plt.colorbar(label='Component Coefficient')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle('Dimensionality Reduction Techniques Comparison', fontsize=16, y=0.98)
        plt.savefig('plots/dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()