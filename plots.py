# -*- coding: utf-8 -*-
"""
机器学习课程作业 - UCI数据集可视化分析
数据集：鸢尾花(Iris) 和 乳腺癌(Breast Cancer)
"""

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 设置高饱和度配色
colors_iris = ['#FF1744', '#00E676', '#2979FF']  # 红、绿、蓝
colors_cancer = ['#FF6F00', '#7C4DFF']  # 橙、紫

print("=" * 60)
print("开始数据可视化分析...")
print("=" * 60)

# ==================== 第一部分：鸢尾花数据集 ====================

# 加载数据
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print("\n【鸢尾花数据集】")
print(f"样本数: {iris_df.shape[0]}, 特征数: {len(iris.feature_names)}")
print(f"类别: {list(iris.target_names)}")

# 图1: 特征箱线图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Dataset - Feature Distribution', fontsize=16, fontweight='bold')

feature_names = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

for idx, (ax, feature, name) in enumerate(zip(axes.flatten(), iris.feature_names, feature_names)):
    # 准备每个类别的数据
    data_plot = [iris_df[iris_df['species'] == i][feature].values for i in range(3)]

    # 绘制箱线图
    bp = ax.boxplot(data_plot, labels=iris.target_names, patch_artist=True, widths=0.6)

    # 设置颜色
    for patch, color in zip(bp['boxes'], colors_iris):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel('Value (cm)', fontweight='bold')
    ax.set_xlabel('Species', fontweight='bold')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('iris_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: iris_boxplot.png")
plt.close()

# 作用说明：
# 展示四个特征在三个物种间的分布差异
# 箱体显示数据的四分位数，横线为中位数
# Petal特征区分度最明显，Setosa明显小于其他两种

# 图2: 特征相关性热力图
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = iris_df[iris.feature_names].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Iris Dataset - Feature Correlation Matrix',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], rotation=45)
ax.set_yticklabels(['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], rotation=0)

plt.tight_layout()
plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: iris_correlation.png")
plt.close()

# 作用说明：
# 展示特征间的线性相关关系
# Petal Length与Petal Width高度正相关(0.96)
# 红色表示正相关，蓝色表示负相关，颜色越深相关性越强

# 图3: 散点图矩阵
iris_plot = iris_df[iris.feature_names + ['species_name']].copy()
iris_plot.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Species']

g = sns.pairplot(iris_plot, hue='Species', palette=colors_iris,
                 height=2.5, aspect=1, diag_kind='kde',
                 plot_kws={'alpha': 0.7, 's': 50})

g.fig.suptitle('Iris Dataset - Pairwise Scatter Plot', fontsize=16, fontweight='bold', y=1.001)
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: iris_pairplot.png")
plt.close()

# 作用说明：
# 展示所有特征两两组合的散点图
# 对角线为各特征的密度分布曲线
# Petal Length vs Petal Width组合分类效果最好
# Setosa(红色)与其他类别分离明显

# 图4: PCA降维可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Iris Dataset - PCA Analysis', fontsize=16, fontweight='bold')

# 数据标准化
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris.data)

# PCA降维
pca = PCA(n_components=4)
iris_pca = pca.fit_transform(iris_scaled)

# 左图：2D投影
for i, (species, color) in enumerate(zip(iris.target_names, colors_iris)):
    mask = iris.target == i
    ax1.scatter(iris_pca[mask, 0], iris_pca[mask, 1],
                c=color, label=species, s=80, alpha=0.8, edgecolors='white')

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
ax1.set_title('2D PCA Projection', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 右图：方差解释率
variance = pca.explained_variance_ratio_ * 100
cumulative = np.cumsum(variance)

x = np.arange(1, 5)
ax2.bar(x, variance, color=colors_iris[0], alpha=0.8, label='Individual')
ax2_twin = ax2.twinx()
ax2_twin.plot(x, cumulative, color=colors_iris[2], marker='o',
              linewidth=3, markersize=10, label='Cumulative')

ax2.set_xlabel('Principal Component', fontweight='bold')
ax2.set_ylabel('Variance Explained (%)', fontweight='bold', color=colors_iris[0])
ax2_twin.set_ylabel('Cumulative (%)', fontweight='bold', color=colors_iris[2])
ax2.set_title('Variance Explained', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'PC{i}' for i in x])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('iris_pca.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: iris_pca.png")
plt.close()

# 作用说明：
# 将4维数据降至2维，保留主要信息
# 前两个主成分解释了约95%的方差
# Setosa在PC1上与其他类别明显分离

# ==================== 第二部分：乳腺癌数据集 ====================

# 加载数据
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
cancer_df['diagnosis'] = cancer.target
cancer_df['diagnosis_name'] = cancer_df['diagnosis'].map({0: 'Malignant', 1: 'Benign'})

print("\n【乳腺癌数据集】")
print(f"样本数: {cancer_df.shape[0]}, 特征数: {len(cancer.feature_names)}")
print(f"类别: {list(cancer.target_names)}")
print(f"类别分布:\n{cancer_df['diagnosis_name'].value_counts()}")

# 选择关键特征
key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean concavity', 'mean concave points']

# 图5: 类别分布和关键特征对比
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
fig.suptitle('Breast Cancer Dataset - Overview', fontsize=16, fontweight='bold')

# 子图1：类别分布柱状图
ax1 = fig.add_subplot(gs[0, 0])
counts = cancer_df['diagnosis_name'].value_counts()
bars = ax1.bar(counts.index, counts.values, color=colors_cancer,
               alpha=0.9, edgecolor='white', linewidth=2)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height,
             f'{int(height)}\n({height / len(cancer_df) * 100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('Count', fontweight='bold')
ax1.set_title('Diagnosis Distribution', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 子图2：类别比例饼图
ax2 = fig.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(counts, labels=counts.index, autopct='%1.1f%%',
                                   colors=colors_cancer, startangle=90,
                                   textprops={'fontweight': 'bold'},
                                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
ax2.set_title('Diagnosis Proportion', fontsize=12, fontweight='bold')

# 子图3：不平衡比率
ax3 = fig.add_subplot(gs[0, 2])
ratio = counts['Benign'] / counts['Malignant']
ax3.text(0.5, 0.5, f'{ratio:.2f} : 1', ha='center', va='center',
         fontsize=28, fontweight='bold', color=colors_cancer[1],
         transform=ax3.transAxes)
ax3.text(0.5, 0.25, '(Benign : Malignant)', ha='center', va='center',
         fontsize=11, style='italic', transform=ax3.transAxes)
ax3.axis('off')
ax3.set_title('Imbalance Ratio', fontsize=12, fontweight='bold')

# 子图4-9：关键特征小提琴图
positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
for pos, feature in zip(positions, key_features):
    ax = fig.add_subplot(gs[pos[0], pos[1]])

    data_m = cancer_df[cancer_df['diagnosis'] == 0][feature]
    data_b = cancer_df[cancer_df['diagnosis'] == 1][feature]

    parts = ax.violinplot([data_m, data_b], positions=[1, 2],
                          showmeans=True, showmedians=True, widths=0.7)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_cancer[i])
        pc.set_alpha(0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Malignant', 'Benign'], fontsize=9)
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(feature.replace('mean ', '').title(), fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.savefig('cancer_overview.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: cancer_overview.png")
plt.close()

# 作用说明：
# 展示类别分布和关键特征的差异
# 良性357例，恶性212例，存在轻度不平衡(1.68:1)
# 小提琴图显示恶性肿瘤在多数特征上数值更大
# 宽度表示数据密度，可观察分布形态

# 图6: 特征相关性
mean_features = [col for col in cancer.feature_names if 'mean' in col][:10]
corr = cancer_df[mean_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Breast Cancer - Feature Correlation (Mean Features)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels([f.replace('mean ', '') for f in mean_features], rotation=45, ha='right')
ax.set_yticklabels([f.replace('mean ', '') for f in mean_features], rotation=0)

plt.tight_layout()
plt.savefig('cancer_correlation.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: cancer_correlation.png")
plt.close()

# 作用说明：
# 展示特征间的相关性
# Radius、Perimeter、Area高度相关(>0.95)，存在几何关系
# Concavity与Concave Points高度相关(0.92)
# 高相关特征存在冗余，可考虑特征选择

# 图7: 关键特征散点图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Breast Cancer - Key Feature Scatter Plots', fontsize=16, fontweight='bold')

feature_pairs = [
    ('mean radius', 'mean texture'),
    ('mean perimeter', 'mean area'),
    ('mean concavity', 'mean concave points'),
    ('mean radius', 'mean concavity'),
    ('mean smoothness', 'mean compactness'),
    ('mean area', 'mean concave points')
]

for ax, (f1, f2) in zip(axes.flatten(), feature_pairs):
    for diag, color, label in zip([0, 1], colors_cancer, ['Malignant', 'Benign']):
        mask = cancer_df['diagnosis'] == diag
        ax.scatter(cancer_df[mask][f1], cancer_df[mask][f2],
                   c=color, label=label, s=40, alpha=0.7, edgecolors='white')

    ax.set_xlabel(f1.replace('mean ', '').title(), fontweight='bold')
    ax.set_ylabel(f2.replace('mean ', '').title(), fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cancer_scatter.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: cancer_scatter.png")
plt.close()

# 作用说明：
# 展示特征两两组合的散点图，观察类别可分性
# Radius vs Texture: 恶性肿瘤半径更大
# Perimeter vs Area: 强线性关系，两类分离明显
# Concavity vs Concave Points: 恶性在两维度都更高，分离效果好

# 图8: PCA降维分析
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Breast Cancer - PCA Analysis', fontsize=16, fontweight='bold')

# 标准化和PCA
scaler = StandardScaler()
cancer_scaled = scaler.fit_transform(cancer.data)
pca = PCA(n_components=10)
cancer_pca = pca.fit_transform(cancer_scaled)

# 左图：2D投影
for diag, color, label in zip([0, 1], colors_cancer, ['Malignant', 'Benign']):
    mask = cancer.target == diag
    ax1.scatter(cancer_pca[mask, 0], cancer_pca[mask, 1],
                c=color, label=label, s=50, alpha=0.7, edgecolors='white')

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
ax1.set_title('2D PCA Projection', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 右图：方差解释率
variance = pca.explained_variance_ratio_[:10] * 100
cumulative = np.cumsum(variance)

x = np.arange(1, 11)
ax2.bar(x, variance, color=colors_cancer[0], alpha=0.8, label='Individual')
ax2_twin = ax2.twinx()
ax2_twin.plot(x, cumulative, color=colors_cancer[1], marker='o',
              linewidth=3, markersize=8, label='Cumulative')

ax2.set_xlabel('Principal Component', fontweight='bold')
ax2.set_ylabel('Variance (%)', fontweight='bold', color=colors_cancer[0])
ax2_twin.set_ylabel('Cumulative (%)', fontweight='bold', color=colors_cancer[1])
ax2.set_title('Variance Explained', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'PC{i}' for i in x])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cancer_pca.png', dpi=300, bbox_inches='tight')
print("✓ 已生成: cancer_pca.png")
plt.close()

# 作用说明：
# 将30维数据降至2维主成分空间
# PC1解释44%方差，PC2解释19%，合计63%
# 两类样本有明显分离趋势，恶性主要在PC1负值区
# 前5个主成分累计解释约85%方差，可有效降维

print("\n" + "=" * 60)
print("所有可视化图表生成完成！")
print("=" * 60)
print("\n生成的图表文件：")
print("【鸢尾花数据集】")
print("  1. iris_boxplot.png - 特征分布箱线图")
print("  2. iris_correlation.png - 特征相关性热力图")
print("  3. iris_pairplot.png - 成对特征散点图矩阵")
print("  4. iris_pca.png - PCA降维分析")
print("\n【乳腺癌数据集】")
print("  5. cancer_overview.png - 类别分布和关键特征对比")
print("  6. cancer_correlation.png - 特征相关性热力图")
print("  7. cancer_scatter.png - 关键特征散点图")
print("  8. cancer_pca.png - PCA降维分析")
print("\n分析完成！共生成8张高质量可视化图表。")
