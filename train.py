import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
from matplotlib import rcParams

warnings.filterwarnings('ignore')

# ==================== 字体设置 (Times New Roman) ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建保存模型和图片的文件夹
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ===================== 1. 加载数据集 =====================
print("=" * 60)
print("加载数据集...")
print("=" * 60)

# 加载鸢尾花数据集
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# 加载乳腺癌数据集
cancer = datasets.load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

# 数据标准化
scaler_iris = StandardScaler()
scaler_cancer = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
X_cancer_scaled = scaler_cancer.fit_transform(X_cancer)

# 划分训练集和测试集
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.3, random_state=42
)
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
    X_cancer_scaled, y_cancer, test_size=0.3, random_state=42
)

print(f"鸢尾花数据集: 训练集 {X_iris_train.shape[0]} 样本, 测试集 {X_iris_test.shape[0]} 样本")
print(f"乳腺癌数据集: 训练集 {X_cancer_train.shape[0]} 样本, 测试集 {X_cancer_test.shape[0]} 样本")

# ===================== 2. 定义模型参数 =====================
print("\n" + "=" * 60)
print("定义模型参数...")
print("=" * 60)

# 10个不同参数的神经网络模型
nn_params = [
    {'hidden_layer_sizes': (10,), 'activation': 'relu', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (20,), 'activation': 'relu', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (50,), 'activation': 'relu', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (10, 10), 'activation': 'relu', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (20, 10), 'activation': 'relu', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (50,), 'activation': 'tanh', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (20,), 'activation': 'logistic', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (30,), 'activation': 'relu', 'learning_rate_init': 0.01,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (30,), 'activation': 'relu', 'learning_rate_init': 0.0001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
    {'hidden_layer_sizes': (15, 15, 15), 'activation': 'relu', 'learning_rate_init': 0.001,
     'max_iter': 2000, 'early_stopping': True, 'n_iter_no_change': 50},
]

# 10个不同参数的SVM模型
svm_params = [
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'linear', 'C': 10.0},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'auto'},
    {'kernel': 'poly', 'C': 1.0, 'degree': 2},
    {'kernel': 'poly', 'C': 1.0, 'degree': 3},
    {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 'scale'},
]

# ===================== 3. 训练模型并保存 =====================
print("\n" + "=" * 60)
print("开始训练模型...")
print("=" * 60)

results = {
    'iris': {'nn': [], 'svm': []},
    'cancer': {'nn': [], 'svm': []}
}

# 训练神经网络模型
print("\n训练神经网络模型...")
for i, params in enumerate(nn_params):
    print(f"  NN模型 {i + 1}/10: {params}")

    # 鸢尾花数据集
    nn_iris = MLPClassifier(random_state=42, **params)
    nn_iris.fit(X_iris_train, y_iris_train)
    y_pred_iris = nn_iris.predict(X_iris_test)

    results['iris']['nn'].append({
        'model': nn_iris,
        'params': params,
        'accuracy': accuracy_score(y_iris_test, y_pred_iris),
        'precision': precision_score(y_iris_test, y_pred_iris, average='weighted'),
        'recall': recall_score(y_iris_test, y_pred_iris, average='weighted'),
        'f1': f1_score(y_iris_test, y_pred_iris, average='weighted'),
    })

    # 乳腺癌数据集
    nn_cancer = MLPClassifier(random_state=42, **params)
    nn_cancer.fit(X_cancer_train, y_cancer_train)
    y_pred_cancer = nn_cancer.predict(X_cancer_test)

    results['cancer']['nn'].append({
        'model': nn_cancer,
        'params': params,
        'accuracy': accuracy_score(y_cancer_test, y_pred_cancer),
        'precision': precision_score(y_cancer_test, y_pred_cancer, average='weighted'),
        'recall': recall_score(y_cancer_test, y_pred_cancer, average='weighted'),
        'f1': f1_score(y_cancer_test, y_pred_cancer, average='weighted'),
    })

    # 保存模型
    joblib.dump(nn_iris, f'models/nn_iris_{i + 1}.pkl')
    joblib.dump(nn_cancer, f'models/nn_cancer_{i + 1}.pkl')

# 训练SVM模型
print("\n训练SVM模型...")
for i, params in enumerate(svm_params):
    print(f"  SVM模型 {i + 1}/10: {params}")

    # 鸢尾花数据集
    svm_iris = SVC(random_state=42, **params)
    svm_iris.fit(X_iris_train, y_iris_train)
    y_pred_iris = svm_iris.predict(X_iris_test)

    results['iris']['svm'].append({
        'model': svm_iris,
        'params': params,
        'accuracy': accuracy_score(y_iris_test, y_pred_iris),
        'precision': precision_score(y_iris_test, y_pred_iris, average='weighted'),
        'recall': recall_score(y_iris_test, y_pred_iris, average='weighted'),
        'f1': f1_score(y_iris_test, y_pred_iris, average='weighted'),
    })

    # 乳腺癌数据集
    svm_cancer = SVC(random_state=42, **params)
    svm_cancer.fit(X_cancer_train, y_cancer_train)
    y_pred_cancer = svm_cancer.predict(X_cancer_test)

    results['cancer']['svm'].append({
        'model': svm_cancer,
        'params': params,
        'accuracy': accuracy_score(y_cancer_test, y_pred_cancer),
        'precision': precision_score(y_cancer_test, y_pred_cancer, average='weighted'),
        'recall': recall_score(y_cancer_test, y_pred_cancer, average='weighted'),
        'f1': f1_score(y_cancer_test, y_pred_cancer, average='weighted'),
    })

    # 保存模型
    joblib.dump(svm_iris, f'models/svm_iris_{i + 1}.pkl')
    joblib.dump(svm_cancer, f'models/svm_cancer_{i + 1}.pkl')

print("\n所有模型训练完成并已保存！")

# ===================== 4. 绘制学习曲线 =====================
print("\n" + "=" * 60)
print("生成学习曲线...")
print("=" * 60)


def plot_learning_curves(models, X_train, y_train, dataset_name, model_type):
    """
    绘制所有模型的学习曲线

    【图表目的】
    - 可视化模型性能如何随训练集大小变化
    - 识别过拟合/欠拟合问题
    - 比较不同模型配置的收敛行为

    【布局说明】
    - 2行 × 5列网格（共10个子图）
    - 每个子图显示一个模型的学习曲线
    - 位置：第一行（模型1-5），第二行（模型6-10）

    【分析方法】
    1. 训练分数（蓝色线）：在训练数据上的性能
       - 应该较高且稳定
       - 如果远高于验证分数：过拟合

    2. 验证分数（珊瑚色线）：在验证数据上的性能
       - 最重要的泛化能力指标
       - 应随训练数据增加而提升
       - 训练/验证分数之间的差距表示过拟合程度

    3. 收敛性：
       - 曲线应在大样本量时趋于平稳
       - 如果仍在上升：更多数据可能有帮助
       - 如果验证分数低：模型复杂度问题

    4. 阴影区域：交叉验证折叠间的标准差
       - 大阴影区域 = 高方差/不稳定
       - 小阴影区域 = 一致的性能
    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle(f'{dataset_name} - {model_type} Learning Curves',
                 fontsize=20, fontweight='bold', y=0.995, family='Times New Roman')
    axes = axes.ravel()
    colors = sns.color_palette("husl", 10)

    for i, result in enumerate(models[:10]):
        model = result['model']

        # 计算学习曲线
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # 绘制训练分数（位置：子图i）
        axes[i].plot(train_sizes, train_mean, 'o-', color=colors[i],
                     linewidth=2.5, markersize=8, label='Training Score', alpha=0.9)
        axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                             alpha=0.15, color=colors[i])

        # 绘制验证分数（位置：子图i）
        axes[i].plot(train_sizes, val_mean, 's--', color='coral',
                     linewidth=2.5, markersize=8, label='Validation Score', alpha=0.9)
        axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                             alpha=0.15, color='coral')

        axes[i].set_xlabel('Training Sample Size', fontsize=11, fontweight='bold', family='Times New Roman')
        axes[i].set_ylabel('Accuracy', fontsize=11, fontweight='bold', family='Times New Roman')
        axes[i].set_title(f'Model {i + 1}', fontsize=13, fontweight='bold', pad=10, family='Times New Roman')
        axes[i].legend(loc='lower right', fontsize=9, prop={'family': 'Times New Roman'})
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig(f'figures/learning_curves_{dataset_name}_{model_type}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: learning_curves_{dataset_name}_{model_type}.png")


# 生成所有学习曲线
plot_learning_curves(results['iris']['nn'], X_iris_train, y_iris_train, 'Iris', 'NN')
plot_learning_curves(results['iris']['svm'], X_iris_train, y_iris_train, 'Iris', 'SVM')
plot_learning_curves(results['cancer']['nn'], X_cancer_train, y_cancer_train, 'BreastCancer', 'NN')
plot_learning_curves(results['cancer']['svm'], X_cancer_train, y_cancer_train, 'BreastCancer', 'SVM')

# ===================== 5. 绘制性能对比图 =====================
print("\n" + "=" * 60)
print("生成性能对比图...")
print("=" * 60)


def plot_performance_comparison(results, dataset_name):
    """
    综合性能对比可视化

    【图表目的】
    - 比较所有模型在所有指标上的表现
    - 识别最佳性能模型
    - 可视化指标之间的关系
    - 显示性能分布模式

    【布局说明】（3行 × 3列）
    第一行：单个模型指标的柱状图
    第二行：所有模型所有指标的热力图
    第三行：最佳模型的雷达图 + 性能分布箱线图

    【各子图分析】

    [第1行, 第1列] NN模型指标柱状图（左上）
    - 显示10个NN模型的4个指标
    - 查找：最高的柱子 = 最佳模型
    - 比较：所有指标是否平衡还是有权衡？

    [第1行, 第2列] SVM模型指标柱状图（中上）
    - 与NN相同，但针对SVM模型
    - 与NN比较，看哪个算法表现更好

    [第1行, 第3列] NN vs SVM准确率折线图（右上）
    - 直接比较各模型编号的准确率
    - 蓝线 = NN，红线 = SVM
    - 查找：哪条线持续更高？

    [第2行, 第1-2列] NN性能热力图（中左，跨2列）
    - 颜色强度 = 性能水平（深色 = 更好）
    - 行 = 指标，列 = 模型
    - 查找：垂直模式（一致的模型）或水平模式（一致的指标）

    [第2行, 第3列] SVM性能热力图（中右）
    - 与NN热力图相同的解释
    - 比较与NN热力图的颜色模式

    [第3行, 第1列] 最佳NN模型雷达图（左下）
    - 五边形显示最佳NN模型的所有4个指标
    - 面积越大 = 整体性能越好
    - 查找：平衡的形状 vs. 尖锐（某些指标弱）

    [第3行, 第2列] 最佳SVM模型雷达图（中下）
    - 与NN雷达图相同
    - 比较形状与NN，看指标差异

    [第3行, 第3列] 性能分布箱线图（右下）
    - 显示NN和SVM的准确率和F1分数分布
    - 箱子 = 四分位距（中间50%）
    - 箱中线 = 中位数
    - 须 = 最小/最大值（排除异常值）
    - 查找：中位数越高 = 平均性能越好
    - 查找：箱子越小 = 性能越一致
    """
    # 提取数据
    nn_acc = [r['accuracy'] for r in results['nn']]
    nn_prec = [r['precision'] for r in results['nn']]
    nn_rec = [r['recall'] for r in results['nn']]
    nn_f1 = [r['f1'] for r in results['nn']]

    svm_acc = [r['accuracy'] for r in results['svm']]
    svm_prec = [r['precision'] for r in results['svm']]
    svm_rec = [r['recall'] for r in results['svm']]
    svm_f1 = [r['f1'] for r in results['svm']]

    # 创建图表
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    fig.suptitle(f'{dataset_name} Dataset - Comprehensive Model Performance Comparison',
                 fontsize=24, fontweight='bold', y=0.98, family='Times New Roman')

    colors_nn = sns.color_palette("Blues_r", 10)
    colors_svm = sns.color_palette("Reds_r", 10)

    # ========== [第1行, 第1列] NN模型指标对比（左上） ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(10)
    width = 0.2
    ax1.bar(x - 1.5 * width, nn_acc, width, label='Accuracy', color='#3498db', alpha=0.9)
    ax1.bar(x - 0.5 * width, nn_prec, width, label='Precision', color='#2ecc71', alpha=0.9)
    ax1.bar(x + 0.5 * width, nn_rec, width, label='Recall', color='#f39c12', alpha=0.9)
    ax1.bar(x + 1.5 * width, nn_f1, width, label='F1 Score', color='#e74c3c', alpha=0.9)
    ax1.set_xlabel('NN Model Number', fontsize=12, fontweight='bold', family='Times New Roman')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold', family='Times New Roman')
    ax1.set_title('Neural Network Performance Comparison', fontsize=14, fontweight='bold',
                  pad=15, family='Times New Roman')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i + 1}' for i in range(10)])
    ax1.legend(fontsize=10, loc='lower right', prop={'family': 'Times New Roman'})
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0.5, 1.05])

    # ========== [第1行, 第2列] SVM模型指标对比（中上） ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - 1.5 * width, svm_acc, width, label='Accuracy', color='#3498db', alpha=0.9)
    ax2.bar(x - 0.5 * width, svm_prec, width, label='Precision', color='#2ecc71', alpha=0.9)
    ax2.bar(x + 0.5 * width, svm_rec, width, label='Recall', color='#f39c12', alpha=0.9)
    ax2.bar(x + 1.5 * width, svm_f1, width, label='F1 Score', color='#e74c3c', alpha=0.9)
    ax2.set_xlabel('SVM Model Number', fontsize=12, fontweight='bold', family='Times New Roman')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold', family='Times New Roman')
    ax2.set_title('Support Vector Machine Performance Comparison', fontsize=14, fontweight='bold',
                  pad=15, family='Times New Roman')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i + 1}' for i in range(10)])
    ax2.legend(fontsize=10, loc='lower right', prop={'family': 'Times New Roman'})
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0.5, 1.05])

    # ========== [第1行, 第3列] NN vs SVM准确率对比（右上） ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(range(1, 11), nn_acc, 'o-', linewidth=3, markersize=10,
             label='Neural Network', color='#3498db', alpha=0.9)
    ax3.plot(range(1, 11), svm_acc, 's-', linewidth=3, markersize=10,
             label='Support Vector Machine', color='#e74c3c', alpha=0.9)
    ax3.set_xlabel('Model Number', fontsize=12, fontweight='bold', family='Times New Roman')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold', family='Times New Roman')
    ax3.set_title('NN vs SVM Accuracy Comparison', fontsize=14, fontweight='bold',
                  pad=15, family='Times New Roman')
    ax3.legend(fontsize=11, loc='lower right', prop={'family': 'Times New Roman'})
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0.5, 1.05])

    # ========== [第2行, 第1-2列] NN性能热力图（中左，跨2列） ==========
    ax4 = fig.add_subplot(gs[1, :2])
    nn_data = np.array([nn_acc, nn_prec, nn_rec, nn_f1])
    im = ax4.imshow(nn_data, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    ax4.set_xticks(np.arange(10))
    ax4.set_yticks(np.arange(4))
    ax4.set_xticklabels([f'NN-{i + 1}' for i in range(10)], family='Times New Roman')
    ax4.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'], family='Times New Roman')
    ax4.set_title('Neural Network Performance Heatmap', fontsize=14, fontweight='bold',
                  pad=15, family='Times New Roman')

    # 添加文本标注
    for i in range(4):
        for j in range(10):
            text = ax4.text(j, i, f'{nn_data[i, j]:.3f}',
                            ha="center", va="center", color="black",
                            fontsize=10, fontweight='bold', family='Times New Roman')

    cbar1 = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar1.set_label('Score', fontsize=11, fontweight='bold', family='Times New Roman')

    # ========== [第2行, 第3列] SVM性能热力图（中右） ==========
    ax5 = fig.add_subplot(gs[1, 2])
    svm_data = np.array([svm_acc, svm_prec, svm_rec, svm_f1])
    im2 = ax5.imshow(svm_data, cmap='YlGnBu', aspect='auto', vmin=0.5, vmax=1.0)
    ax5.set_xticks(np.arange(10))
    ax5.set_yticks(np.arange(4))
    ax5.set_xticklabels([f'SVM-{i + 1}' for i in range(10)], rotation=45, ha='right',
                        family='Times New Roman')
    ax5.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'], family='Times New Roman')
    ax5.set_title('SVM Performance Heatmap', fontsize=14, fontweight='bold',
                  pad=15, family='Times New Roman')

    # 添加文本标注
    for i in range(4):
        for j in range(10):
            text = ax5.text(j, i, f'{svm_data[i, j]:.3f}',
                            ha="center", va="center", color="black",
                            fontsize=9, fontweight='bold', family='Times New Roman')

    cbar2 = plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)
    cbar2.set_label('Score', fontsize=11, fontweight='bold', family='Times New Roman')

    # ========== [第3行, 第1列] 最佳NN模型雷达图（左下） ==========
    ax6 = fig.add_subplot(gs[2, 0], projection='polar')
    best_nn_idx = np.argmax(nn_acc)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [nn_acc[best_nn_idx], nn_prec[best_nn_idx],
              nn_rec[best_nn_idx], nn_f1[best_nn_idx]]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax6.plot(angles, values, 'o-', linewidth=3, color='#3498db', markersize=10, alpha=0.9)
    ax6.fill(angles, values, alpha=0.25, color='#3498db')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=11, family='Times New Roman')
    ax6.set_ylim(0, 1)
    ax6.set_title(f'Best NN Model (Model {best_nn_idx + 1})',
                  fontsize=14, fontweight='bold', pad=20, family='Times New Roman')
    ax6.grid(True, linestyle='--', alpha=0.5)

    # ========== [第3行, 第2列] 最佳SVM模型雷达图（中下） ==========
    ax7 = fig.add_subplot(gs[2, 1], projection='polar')
    best_svm_idx = np.argmax(svm_acc)
    values_svm = [svm_acc[best_svm_idx], svm_prec[best_svm_idx],
                  svm_rec[best_svm_idx], svm_f1[best_svm_idx]]
    values_svm += values_svm[:1]

    ax7.plot(angles, values_svm, 's-', linewidth=3, color='#e74c3c', markersize=10, alpha=0.9)
    ax7.fill(angles, values_svm, alpha=0.25, color='#e74c3c')
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories, fontsize=11, family='Times New Roman')
    ax7.set_ylim(0, 1)
    ax7.set_title(f'Best SVM Model (Model {best_svm_idx + 1})',
                  fontsize=14, fontweight='bold', pad=20, family='Times New Roman')
    ax7.grid(True, linestyle='--', alpha=0.5)

    # ========== [第3行, 第3列] 性能分布箱线图（右下） ==========
    ax8 = fig.add_subplot(gs[2, 2])
    box_data = [nn_acc, svm_acc, nn_f1, svm_f1]
    bp = ax8.boxplot(box_data, labels=['NN\nAccuracy', 'SVM\nAccuracy', 'NN\nF1 Score', 'SVM\nF1 Score'],
                     patch_artist=True, widths=0.6)

    colors_box = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)

    ax8.set_ylabel('Score', fontsize=12, fontweight='bold', family='Times New Roman')
    ax8.set_title('Model Performance Distribution', fontsize=14, fontweight='bold',
                  pad=15, family='Times New Roman')
    ax8.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax8.set_ylim([0.5, 1.05])

    # 设置刻度标签字体
    for label in ax8.get_xticklabels():
        label.set_fontfamily('Times New Roman')

    plt.savefig(f'figures/performance_comparison_{dataset_name}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: performance_comparison_{dataset_name}.png")


# 生成性能对比图
plot_performance_comparison(results['iris'], 'Iris')
plot_performance_comparison(results['cancer'], 'BreastCancer')

# ===================== 6. 生成综合对比图 =====================
print("\n" + "=" * 60)
print("生成综合对比图...")
print("=" * 60)


def plot_comprehensive_comparison():
    """
    跨数据集综合对比

    【图表目的】
    - 比较两个数据集上的模型性能
    - 识别哪些模型泛化能力更好
    - 显示数据集特定的性能模式

    【布局说明】（2行 × 4列）
    - 第1行：鸢尾花数据集分析
    - 第2行：乳腺癌数据集分析
    - 每行有4种分析类型

    【各子图分析】

    [第1列] 准确率柱状图（左）
    - 直接比较每个模型编号的NN vs SVM准确率
    - 蓝色柱 = NN，红色柱 = SVM
    - 查找：哪种颜色占主导？（更好的算法）
    - 查找：一致的模式还是因模型而异？

    [第2列] F1分数柱状图（中左）
    - 与准确率相同，但针对F1分数
    - F1平衡了精确率和召回率
    - 对不平衡数据集很重要
    - 与准确率比较：排名是否一致？

    [第3列] 平均性能柱状图（中右）
    - 显示所有4个指标的平均值
    - 给出算法的整体比较
    - 查找：哪个算法持续更好？
    - 查找：差异是显著的还是边际的？

    [第4列] 准确率 vs F1散点图（右）
    - 每个点 = 一个模型
    - X轴 = 准确率，Y轴 = F1分数
    - 圆圈 = NN，方块 = SVM
    - 内部数字 = 模型编号
    - 查找：右上角的点 = 最佳模型
    - 查找：聚类模式（相似性能）
    - 查找：异常值（异常好/坏的模型）

    【跨数据集比较】
    - 比较第1行 vs 第2行以查看：
      1. 哪个数据集更容易？（更高分数）
      2. 哪个算法在数据集间更一致？
      3. 最佳模型编号在数据集间是否相同？
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Iris vs Breast Cancer Dataset - Comprehensive Model Performance Comparison',
                 fontsize=22, fontweight='bold', y=0.995, family='Times New Roman')

    datasets = [('iris', 'Iris'), ('cancer', 'Breast Cancer')]

    for row, (dataset_key, dataset_name) in enumerate(datasets):
        # 提取数据
        nn_acc = [r['accuracy'] for r in results[dataset_key]['nn']]
        svm_acc = [r['accuracy'] for r in results[dataset_key]['svm']]
        nn_f1 = [r['f1'] for r in results[dataset_key]['nn']]
        svm_f1 = [r['f1'] for r in results[dataset_key]['svm']]

        # ========== [第1列] 准确率对比（左） ==========
        ax = axes[row, 0]
        x = np.arange(10)
        width = 0.35
        ax.bar(x - width / 2, nn_acc, width, label='Neural Network', color='#3498db', alpha=0.9)
        ax.bar(x + width / 2, svm_acc, width, label='Support Vector Machine', color='#e74c3c', alpha=0.9)
        ax.set_xlabel('Model Number', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_title(f'{dataset_name} - Accuracy Comparison', fontsize=13, fontweight='bold',
                     family='Times New Roman')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i + 1}' for i in range(10)])
        ax.legend(fontsize=10, prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0.5, 1.05])

        # ========== [第2列] F1分数对比（中左） ==========
        ax = axes[row, 1]
        ax.bar(x - width / 2, nn_f1, width, label='Neural Network', color='#2ecc71', alpha=0.9)
        ax.bar(x + width / 2, svm_f1, width, label='Support Vector Machine', color='#f39c12', alpha=0.9)
        ax.set_xlabel('Model Number', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_title(f'{dataset_name} - F1 Score Comparison', fontsize=13, fontweight='bold',
                     family='Times New Roman')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i + 1}' for i in range(10)])
        ax.legend(fontsize=10, prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0.5, 1.05])

        # ========== [第3列] 平均性能对比（中右） ==========
        ax = axes[row, 2]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        nn_means = [
            np.mean([r['accuracy'] for r in results[dataset_key]['nn']]),
            np.mean([r['precision'] for r in results[dataset_key]['nn']]),
            np.mean([r['recall'] for r in results[dataset_key]['nn']]),
            np.mean([r['f1'] for r in results[dataset_key]['nn']])
        ]
        svm_means = [
            np.mean([r['accuracy'] for r in results[dataset_key]['svm']]),
            np.mean([r['precision'] for r in results[dataset_key]['svm']]),
            np.mean([r['recall'] for r in results[dataset_key]['svm']]),
            np.mean([r['f1'] for r in results[dataset_key]['svm']])
        ]

        x_pos = np.arange(len(metrics))
        ax.bar(x_pos - width / 2, nn_means, width, label='Neural Network', color='#9b59b6', alpha=0.9)
        ax.bar(x_pos + width / 2, svm_means, width, label='Support Vector Machine', color='#e67e22', alpha=0.9)
        ax.set_xlabel('Evaluation Metric', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Average Score', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_title(f'{dataset_name} - Average Performance Comparison', fontsize=13, fontweight='bold',
                     family='Times New Roman')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=15, ha='right', family='Times New Roman')
        ax.legend(fontsize=10, prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0.5, 1.05])

        # ========== [第4列] 准确率 vs F1散点图（右） ==========
        ax = axes[row, 3]
        ax.scatter(nn_acc, nn_f1, s=200, alpha=0.7, c='#3498db',
                   edgecolors='black', linewidth=2, label='Neural Network', marker='o')
        ax.scatter(svm_acc, svm_f1, s=200, alpha=0.7, c='#e74c3c',
                   edgecolors='black', linewidth=2, label='Support Vector Machine', marker='s')

        # 添加模型编号标注
        for i in range(10):
            ax.annotate(f'{i + 1}', (nn_acc[i], nn_f1[i]),
                        fontsize=9, ha='center', va='center', fontweight='bold',
                        family='Times New Roman')
            ax.annotate(f'{i + 1}', (svm_acc[i], svm_f1[i]),
                        fontsize=9, ha='center', va='center', fontweight='bold',
                        family='Times New Roman')

        ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold', family='Times New Roman')
        ax.set_title(f'{dataset_name} - Accuracy vs F1 Score', fontsize=13, fontweight='bold',
                     family='Times New Roman')
        ax.legend(fontsize=10, loc='lower right', prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0.5, 1.05])
        ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig('figures/comprehensive_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  已保存: comprehensive_comparison.png")


plot_comprehensive_comparison()

# ===================== 7. 生成结果报告 =====================
print("\n" + "=" * 60)
print("生成结果报告...")
print("=" * 60)


def generate_report():
    """生成文本报告"""
    report = []
    report.append("=" * 80)
    report.append("Machine Learning Model Performance Evaluation Report")
    report.append("=" * 80)
    report.append("")

    for dataset_name, dataset_key in [('Iris Dataset', 'iris'), ('Breast Cancer Dataset', 'cancer')]:
        report.append(f"\n{'=' * 80}")
        report.append(f"{dataset_name}")
        report.append(f"{'=' * 80}")

        # 神经网络结果
        report.append("\n[Neural Network Models]")
        report.append("-" * 80)
        report.append(f"{'Model':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        report.append("-" * 80)

        for i, r in enumerate(results[dataset_key]['nn']):
            report.append(f"NN-{i + 1:<5} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                          f"{r['recall']:<12.4f} {r['f1']:<12.4f}")

        nn_acc_mean = np.mean([r['accuracy'] for r in results[dataset_key]['nn']])
        nn_acc_std = np.std([r['accuracy'] for r in results[dataset_key]['nn']])
        best_nn = max(results[dataset_key]['nn'], key=lambda x: x['accuracy'])
        best_nn_idx = results[dataset_key]['nn'].index(best_nn)

        report.append("-" * 80)
        report.append(f"Average Accuracy: {nn_acc_mean:.4f} ± {nn_acc_std:.4f}")
        report.append(f"Best Model: NN-{best_nn_idx + 1}, Accuracy: {best_nn['accuracy']:.4f}")

        # SVM结果
        report.append("\n[Support Vector Machine Models]")
        report.append("-" * 80)
        report.append(f"{'Model':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        report.append("-" * 80)

        for i, r in enumerate(results[dataset_key]['svm']):
            report.append(f"SVM-{i + 1:<5} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                          f"{r['recall']:<12.4f} {r['f1']:<12.4f}")

        svm_acc_mean = np.mean([r['accuracy'] for r in results[dataset_key]['svm']])
        svm_acc_std = np.std([r['accuracy'] for r in results[dataset_key]['svm']])
        best_svm = max(results[dataset_key]['svm'], key=lambda x: x['accuracy'])
        best_svm_idx = results[dataset_key]['svm'].index(best_svm)

        report.append("-" * 80)
        report.append(f"Average Accuracy: {svm_acc_mean:.4f} ± {svm_acc_std:.4f}")
        report.append(f"Best Model: SVM-{best_svm_idx + 1}, Accuracy: {best_svm['accuracy']:.4f}")

        # 总结
        report.append("\n[Summary]")
        report.append("-" * 80)
        if nn_acc_mean > svm_acc_mean:
            report.append(f"On {dataset_name}, Neural Network performs better overall")
        else:
            report.append(f"On {dataset_name}, Support Vector Machine performs better overall")
        report.append("")

    report.append("\n" + "=" * 80)
    report.append("Report Generation Complete!")
    report.append("=" * 80)

    # 保存报告
    with open('figures/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 打印报告
    for line in report:
        print(line)


generate_report()

# ===================== 8. 打印总结信息 =====================
print("\n" + "=" * 60)
print("任务完成总结")
print("=" * 60)
print(f"✓ 已训练 20 个神经网络模型（10个参数配置 × 2个数据集）")
print(f"✓ 已训练 20 个支持向量机模型（10个参数配置 × 2个数据集）")
print(f"✓ 所有模型已保存到 'models/' 文件夹")
print(f"✓ 已生成 4 张学习曲线图")
print(f"✓ 已生成 2 张详细性能对比图")
print(f"✓ 已生成 1 张综合对比图")
print(f"✓ 已生成性能评估报告")
print(f"\n所有图表保存在 'figures/' 文件夹中")
print("=" * 60)
