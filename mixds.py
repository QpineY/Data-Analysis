import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from itertools import combinations
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=" * 80)
print("DS证据理论模型融合系统")
print("=" * 80)


# ===================== 1. DS证据理论融合器 =====================
class DSTheoryFusion:
    """
    Dempster-Shafer证据理论融合器

    【核心思想】
    - 将每个分类器的预测转换为基本概率分配(BPA)
    - 使用Dempster组合规则融合多个证据源
    - 基于融合后的信度做出最终决策

    【优势】
    - 能处理不确定性和冲突证据
    - 比简单投票更加稳健
    - 考虑了模型置信度
    """

    def __init__(self, models, class_labels):
        """
        初始化融合器

        参数:
            models: 训练好的模型列表
            class_labels: 类别标签数组
        """
        self.models = models
        self.class_labels = class_labels
        self.n_classes = len(np.unique(class_labels))

    def compute_bpa(self, model, X, y_true):
        """
        计算基本概率分配(Basic Probability Assignment)

        【方法】
        1. 使用模型在训练集上的准确率作为可信度
        2. 对于预测类别c: m(c) = accuracy
        3. 对于不确定性: m(Θ) = 1 - accuracy

        参数:
            model: 单个分类器
            X: 输入特征
            y_true: 真实标签

        返回:
            bpa_matrix: shape (n_samples, n_classes+1)
                       最后一列是不确定性 m(Θ)
        """
        # 计算模型可信度（基于训练集准确率）
        y_pred_train = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred_train)

        # 对测试集进行预测
        predictions = model.predict(X)

        # 初始化BPA矩阵
        n_samples = len(predictions)
        bpa_matrix = np.zeros((n_samples, self.n_classes + 1))

        # 为每个样本分配概率
        for i, pred in enumerate(predictions):
            # 预测类别的信度 = 模型准确率
            bpa_matrix[i, pred] = accuracy
            # 不确定性
            bpa_matrix[i, -1] = 1 - accuracy

        return bpa_matrix

    def dempster_combine(self, bpa1, bpa2):
        """
        Dempster组合规则

        【公式】
        m(A) = Σ[m1(B) × m2(C)] / (1 - K)
        其中 B∩C = A, K是冲突系数

        参数:
            bpa1, bpa2: 两个BPA向量 shape (n_classes+1,)

        返回:
            combined_bpa: 融合后的BPA向量
        """
        n_classes = len(bpa1) - 1
        combined = np.zeros(n_classes + 1)

        # 计算冲突系数K
        conflict = 0.0
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    conflict += bpa1[i] * bpa2[j]

        # 避免完全冲突
        if conflict >= 0.9999:
            conflict = 0.9999

        # Dempster组合规则
        # 1. 相同类别的组合
        for i in range(n_classes):
            combined[i] = bpa1[i] * bpa2[i]
            # 一个类别 × 不确定性
            combined[i] += bpa1[i] * bpa2[-1]
            combined[i] += bpa1[-1] * bpa2[i]

        # 2. 不确定性的组合
        combined[-1] = bpa1[-1] * bpa2[-1]

        # 归一化（除以1-K）
        if conflict < 1.0:
            combined = combined / (1 - conflict)

        return combined

    def fuse_predictions(self, X_test, X_train, y_train):
        """
        融合所有模型的预测

        【流程】
        1. 计算每个模型的BPA
        2. 使用Dempster规则逐步融合
        3. 选择信度最高的类别

        参数:
            X_test: 测试集特征
            X_train: 训练集特征（用于计算可信度）
            y_train: 训练集标签

        返回:
            final_predictions: 融合后的预测
            fusion_confidence: 融合置信度
        """
        n_samples = X_test.shape[0]
        n_models = len(self.models)

        # 存储所有模型的BPA
        all_bpas = []
        for model in self.models:
            bpa = self.compute_bpa(model, X_train, y_train)
            # 对测试集计算BPA
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred_train)

            test_bpa = np.zeros((n_samples, self.n_classes + 1))
            for i, pred in enumerate(y_pred_test):
                test_bpa[i, pred] = accuracy
                test_bpa[i, -1] = 1 - accuracy

            all_bpas.append(test_bpa)

        # 逐步融合所有BPA
        fused_bpa = all_bpas[0].copy()
        for i in range(1, n_models):
            for sample_idx in range(n_samples):
                fused_bpa[sample_idx] = self.dempster_combine(
                    fused_bpa[sample_idx],
                    all_bpas[i][sample_idx]
                )

        # 基于融合BPA做出最终决策
        final_predictions = np.argmax(fused_bpa[:, :-1], axis=1)
        fusion_confidence = np.max(fused_bpa[:, :-1], axis=1)

        return final_predictions, fusion_confidence, fused_bpa


# ===================== 2. 加载已训练的模型 =====================
print("\n" + "=" * 80)
print("加载已训练的模型...")
print("=" * 80)

# 加载模型
models_dict = {
    'iris': {'nn': [], 'svm': []},
    'cancer': {'nn': [], 'svm': []}
}

for dataset in ['iris', 'cancer']:
    for model_type in ['nn', 'svm']:
        for i in range(1, 11):
            model_path = f'models/{model_type}_{dataset}_{i}.pkl'
            model = joblib.load(model_path)
            models_dict[dataset][model_type].append(model)

print(f"✓ 已加载 40 个模型")

# 重新加载数据集
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 鸢尾花数据集
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.3, random_state=42
)

# 乳腺癌数据集
cancer = datasets.load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target
scaler_cancer = StandardScaler()
X_cancer_scaled = scaler_cancer.fit_transform(X_cancer)
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
    X_cancer_scaled, y_cancer, test_size=0.3, random_state=42
)

print(f"✓ 数据集已加载")

# ===================== 3. 执行DS融合 =====================
print("\n" + "=" * 80)
print("执行DS证据理论融合...")
print("=" * 80)

fusion_results = {}

for dataset_name, (X_train, X_test, y_train, y_test) in [
    ('iris', (X_iris_train, X_iris_test, y_iris_train, y_iris_test)),
    ('cancer', (X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test))
]:
    print(f"\n处理 {dataset_name.upper()} 数据集...")
    fusion_results[dataset_name] = {}

    for model_type in ['nn', 'svm']:
        print(f"  融合 {model_type.upper()} 模型...")

        # 创建融合器
        models = models_dict[dataset_name][model_type]
        fusion = DSTheoryFusion(models, y_train)

        # 执行融合
        fused_pred, confidence, bpa_matrix = fusion.fuse_predictions(
            X_test, X_train, y_train
        )

        # 计算融合后的性能指标
        fusion_acc = accuracy_score(y_test, fused_pred)
        fusion_prec = precision_score(y_test, fused_pred, average='weighted')
        fusion_rec = recall_score(y_test, fused_pred, average='weighted')
        fusion_f1 = f1_score(y_test, fused_pred, average='weighted')

        # 计算单个模型的平均性能（用于对比）
        individual_accs = []
        individual_precs = []
        individual_recs = []
        individual_f1s = []
        individual_preds = []

        for model in models:
            pred = model.predict(X_test)
            individual_preds.append(pred)
            individual_accs.append(accuracy_score(y_test, pred))
            individual_precs.append(precision_score(y_test, pred, average='weighted'))
            individual_recs.append(recall_score(y_test, pred, average='weighted'))
            individual_f1s.append(f1_score(y_test, pred, average='weighted'))

        # 简单投票融合（用于对比）
        individual_preds_array = np.array(individual_preds)  # shape: (10, n_samples)
        voting_pred = []
        for i in range(len(y_test)):
            # 获取第i个样本的所有模型预测
            votes = [individual_preds[j][i] for j in range(10)]
            # 统计投票
            vote_counts = np.bincount(votes)
            # 选择得票最多的类别
            voting_pred.append(np.argmax(vote_counts))
        voting_pred = np.array(voting_pred)
        voting_acc = accuracy_score(y_test, voting_pred)

        # 保存结果
        fusion_results[dataset_name][model_type] = {
            'fused_predictions': fused_pred,
            'confidence': confidence,
            'bpa_matrix': bpa_matrix,
            'fusion_accuracy': fusion_acc,
            'fusion_precision': fusion_prec,
            'fusion_recall': fusion_rec,
            'fusion_f1': fusion_f1,
            'individual_accuracies': individual_accs,
            'individual_precisions': individual_precs,
            'individual_recalls': individual_recs,
            'individual_f1s': individual_f1s,
            'avg_individual_acc': np.mean(individual_accs),
            'best_individual_acc': np.max(individual_accs),
            'voting_accuracy': voting_acc,
            'confusion_matrix': confusion_matrix(y_test, fused_pred)
        }

        print(f"    DS融合准确率: {fusion_acc:.4f}")
        print(f"    平均单模型准确率: {np.mean(individual_accs):.4f}")
        print(f"    最佳单模型准确率: {np.max(individual_accs):.4f}")
        print(f"    简单投票准确率: {voting_acc:.4f}")
        print(f"    性能提升: {(fusion_acc - np.mean(individual_accs)) * 100:.2f}%")

print("\n✓ DS融合完成！")

# ===================== 4. 可视化：融合效果对比 =====================
print("\n" + "=" * 80)
print("生成融合效果对比图...")
print("=" * 80)


def plot_fusion_comparison():
    """
    融合方法综合对比

    【图表布局】(2行 × 3列)
    第1行: 鸢尾花数据集
    第2行: 乳腺癌数据集

    每行包含:
    [列1] NN模型融合对比
    [列2] SVM模型融合对比
    [列3] 融合方法性能对比
    """

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('DS Evidence Theory Fusion - Comprehensive Performance Comparison',
                 fontsize=22, fontweight='bold', y=0.995)

    datasets = [('iris', 'Iris', y_iris_test),
                ('cancer', 'Breast Cancer', y_cancer_test)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for row, (dataset_key, dataset_name, y_test) in enumerate(datasets):

        for col, model_type in enumerate(['nn', 'svm']):
            ax = axes[row, col]

            results = fusion_results[dataset_key][model_type]

            # 准备数据
            individual_accs = results['individual_accuracies']
            fusion_acc = results['fusion_accuracy']
            voting_acc = results['voting_accuracy']
            avg_acc = results['avg_individual_acc']
            best_acc = results['best_individual_acc']

            # 绘制单个模型准确率
            x = np.arange(1, 11)
            bars = ax.bar(x, individual_accs, alpha=0.7, color=colors[col],
                          edgecolor='black', linewidth=1.5, label='Individual Models')

            # 添加对比线
            ax.axhline(y=fusion_acc, color='#2ecc71', linestyle='-',
                       linewidth=3, label=f'DS Fusion ({fusion_acc:.4f})', alpha=0.9)
            ax.axhline(y=voting_acc, color='#f39c12', linestyle='--',
                       linewidth=2.5, label=f'Simple Voting ({voting_acc:.4f})', alpha=0.9)
            ax.axhline(y=avg_acc, color='gray', linestyle=':',
                       linewidth=2, label=f'Average ({avg_acc:.4f})', alpha=0.7)

            # 标注最佳单模型
            best_idx = np.argmax(individual_accs)
            ax.scatter([best_idx + 1], [best_acc], s=300, color='red',
                       marker='*', zorder=5, label=f'Best Single ({best_acc:.4f})')

            ax.set_xlabel('Model Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset_name} - {model_type.upper()} Fusion Comparison',
                         fontsize=13, fontweight='bold', pad=10)
            ax.set_xticks(x)
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim([0.5, 1.05])

        # 第3列：融合方法对比
        ax = axes[row, 2]

        methods = ['Avg\nIndividual', 'Best\nIndividual', 'Simple\nVoting', 'DS\nFusion']
        nn_scores = [
            fusion_results[dataset_key]['nn']['avg_individual_acc'],
            fusion_results[dataset_key]['nn']['best_individual_acc'],
            fusion_results[dataset_key]['nn']['voting_accuracy'],
            fusion_results[dataset_key]['nn']['fusion_accuracy']
        ]
        svm_scores = [
            fusion_results[dataset_key]['svm']['avg_individual_acc'],
            fusion_results[dataset_key]['svm']['best_individual_acc'],
            fusion_results[dataset_key]['svm']['voting_accuracy'],
            fusion_results[dataset_key]['svm']['fusion_accuracy']
        ]

        x_pos = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x_pos - width / 2, nn_scores, width, label='Neural Network',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width / 2, svm_scores, width, label='SVM',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.set_xlabel('Fusion Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name} - Fusion Methods Comparison',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig('figures/ds_fusion_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: ds_fusion_comparison.png")


plot_fusion_comparison()


# ===================== 5. 可视化：详细性能指标对比 =====================
def plot_detailed_metrics():
    """
    详细性能指标对比（4个指标）
    """

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('DS Fusion - Detailed Performance Metrics Comparison',
                 fontsize=22, fontweight='bold', y=0.995)

    # 定义指标信息
    metrics_info = [
        ('Accuracy', 'individual_accuracies', 'fusion_accuracy'),
        ('Precision', 'individual_precisions', 'fusion_precision'),
        ('Recall', 'individual_recalls', 'fusion_recall'),
        ('F1 Score', 'individual_f1s', 'fusion_f1')
    ]

    for idx, (metric_name, individual_key, fusion_key) in enumerate(metrics_info):
        ax = axes[idx // 2, idx % 2]

        # 准备数据
        data_to_plot = []
        labels = []
        colors_list = []

        for dataset in ['iris', 'cancer']:
            for model_type in ['nn', 'svm']:
                try:
                    # 单个模型的指标
                    individual_scores = fusion_results[dataset][model_type][individual_key]
                    avg_score = np.mean(individual_scores)

                    # 融合后的指标
                    fusion_score = fusion_results[dataset][model_type][fusion_key]

                    data_to_plot.extend([avg_score, fusion_score])
                    labels.extend([
                        f'{dataset.capitalize()}\n{model_type.upper()}\nAvg',
                        f'{dataset.capitalize()}\n{model_type.upper()}\nFusion'
                    ])
                    colors_list.extend(['lightgray', '#2ecc71'])

                except KeyError as e:
                    print(f"警告: 找不到键 {e}，跳过 {dataset}-{model_type}-{metric_name}")
                    continue

        # 绘制柱状图
        x_pos = np.arange(len(data_to_plot))
        bars = ax.bar(x_pos, data_to_plot, color=colors_list, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, data_to_plot)):
            height = bar.get_height()
            # 计算提升百分比（仅对融合结果）
            if i % 2 == 1 and i > 0:  # 融合结果
                improvement = ((score - data_to_plot[i - 1]) / data_to_plot[i - 1]) * 100
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{score:.3f}\n(+{improvement:.1f}%)',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color='green' if improvement > 0 else 'red')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{score:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison: Individual vs DS Fusion',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0.5, 1.05])

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', label='Average Individual'),
            Patch(facecolor='#2ecc71', edgecolor='black', label='DS Fusion')
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig('figures/ds_fusion_detailed_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: ds_fusion_detailed_metrics.png")


plot_detailed_metrics()


# ===================== 6. 可视化：混淆矩阵 =====================
def plot_confusion_matrices():
    """
    绘制融合后的混淆矩阵
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('DS Fusion - Confusion Matrices',
                 fontsize=22, fontweight='bold', y=0.995)

    datasets_info = [
        ('iris', 'Iris', ['Setosa', 'Versicolor', 'Virginica']),
        ('cancer', 'Breast Cancer', ['Malignant', 'Benign'])
    ]

    for row, (dataset_key, dataset_name, class_names) in enumerate(datasets_info):
        for col, model_type in enumerate(['nn', 'svm']):
            ax = axes[row, col]

            cm = fusion_results[dataset_key][model_type]['confusion_matrix']

            # 归一化混淆矩阵
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # 绘制热力图
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlOrRd')

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Count', fontsize=11, fontweight='bold')

            # 设置刻度
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, fontsize=11)
            ax.set_yticklabels(class_names, fontsize=11)

            # 添加文本标注
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                            ha="center", va="center",
                            color="white" if cm_normalized[i, j] > thresh else "black",
                            fontsize=12, fontweight='bold')

            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset_name} - {model_type.upper()} DS Fusion',
                         fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('figures/ds_fusion_confusion_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: ds_fusion_confusion_matrices.png")


plot_confusion_matrices()


# ===================== 7. 可视化：置信度分析 =====================
def plot_confidence_analysis():
    """
    分析DS融合的置信度分布
    """

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('DS Fusion - Confidence Analysis',
                 fontsize=22, fontweight='bold', y=0.995)

    for row, dataset in enumerate(['iris', 'cancer']):
        for col, model_type in enumerate(['nn', 'svm']):
            ax = axes[row, col]

            confidence = fusion_results[dataset][model_type]['confidence']

            # 绘制置信度分布直方图
            n, bins, patches = ax.hist(confidence, bins=20, alpha=0.7,
                                       color='#3498db', edgecolor='black', linewidth=1.5)

            # 根据置信度着色
            cm = plt.cm.RdYlGn
            for i, patch in enumerate(patches):
                patch.set_facecolor(cm(bins[i]))

            # 添加统计信息
            mean_conf = np.mean(confidence)
            median_conf = np.median(confidence)
            min_conf = np.min(confidence)
            max_conf = np.max(confidence)

            ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2.5,
                       label=f'Mean: {mean_conf:.3f}')
            ax.axvline(median_conf, color='green', linestyle='--', linewidth=2.5,
                       label=f'Median: {median_conf:.3f}')

            # 添加文本框
            textstr = f'Min: {min_conf:.3f}\nMax: {max_conf:.3f}\nStd: {np.std(confidence):.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, fontweight='bold')

            ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset.capitalize()} - {model_type.upper()} Confidence Distribution',
                         fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig('figures/ds_fusion_confidence_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: ds_fusion_confidence_analysis.png")


plot_confidence_analysis()


# ===================== 8. 可视化：性能提升雷达图 =====================
def plot_improvement_radar():
    """
    使用雷达图展示DS融合相对于单模型的性能提升
    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(projection='polar'))
    fig.suptitle('DS Fusion Performance Improvement - Radar Chart',
                 fontsize=20, fontweight='bold', y=1.02)

    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 修正：创建指标名称到键名的映射（正确的复数形式）
    metric_key_mapping = {
        'accuracy': 'individual_accuracies',
        'precision': 'individual_precisions',
        'recall': 'individual_recalls',
        'f1': 'individual_f1s'
    }

    for idx, dataset in enumerate(['iris', 'cancer']):
        ax = axes[idx]

        for model_type, color, marker in [('nn', '#3498db', 'o'), ('svm', '#e74c3c', 's')]:
            # 计算提升百分比
            improvements = []
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                fusion_score = fusion_results[dataset][model_type][f'fusion_{metric}']
                # 使用正确的键名获取单个模型的指标
                individual_key = metric_key_mapping[metric]
                avg_individual = np.mean(fusion_results[dataset][model_type][individual_key])
                improvement = ((fusion_score - avg_individual) / avg_individual) * 100
                improvements.append(improvement)

            improvements += improvements[:1]

            ax.plot(angles, improvements, marker, linewidth=3, markersize=10,
                    label=f'{model_type.upper()}', color=color, alpha=0.8)
            ax.fill(angles, improvements, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{dataset.capitalize()} Dataset\nPerformance Improvement (%)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('figures/ds_fusion_improvement_radar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: ds_fusion_improvement_radar.png")


plot_improvement_radar()

# ===================== 9. 生成DS融合报告 =====================
print("\n" + "=" * 80)
print("生成DS融合分析报告...")
print("=" * 80)


def generate_fusion_report():
    """生成详细的融合分析报告"""

    report = []
    report.append("=" * 100)
    report.append("DS EVIDENCE THEORY FUSION - COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("")

    for dataset_name, dataset_key in [('IRIS DATASET', 'iris'),
                                      ('BREAST CANCER DATASET', 'cancer')]:
        report.append("\n" + "=" * 100)
        report.append(f"{dataset_name}")
        report.append("=" * 100)

        for model_name, model_type in [('NEURAL NETWORK', 'nn'),
                                       ('SUPPORT VECTOR MACHINE', 'svm')]:
            report.append(f"\n{'─' * 100}")
            report.append(f"{model_name} MODELS")
            report.append(f"{'─' * 100}")

            results = fusion_results[dataset_key][model_type]

            # 单个模型性能
            report.append("\n[Individual Model Performance]")
            report.append(f"{'Model':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
            report.append("─" * 100)

            for i in range(10):
                report.append(
                    f"Model-{i + 1:<4} "
                    f"{results['individual_accuracies'][i]:<12.4f} "
                    f"{results['individual_precisions'][i]:<12.4f} "
                    f"{results['individual_recalls'][i]:<12.4f} "
                    f"{results['individual_f1s'][i]:<12.4f}"
                )

            report.append("─" * 100)
            report.append(f"{'Average:':<10} "
                          f"{results['avg_individual_acc']:<12.4f} "
                          f"{np.mean(results['individual_precisions']):<12.4f} "
                          f"{np.mean(results['individual_recalls']):<12.4f} "
                          f"{np.mean(results['individual_f1s']):<12.4f}")
            report.append(f"{'Best:':<10} "
                          f"{results['best_individual_acc']:<12.4f} "
                          f"{np.max(results['individual_precisions']):<12.4f} "
                          f"{np.max(results['individual_recalls']):<12.4f} "
                          f"{np.max(results['individual_f1s']):<12.4f}")

            # 融合结果
            report.append("\n[Fusion Results]")
            report.append("─" * 100)
            report.append(f"DS Fusion Accuracy:      {results['fusion_accuracy']:.4f}")
            report.append(f"DS Fusion Precision:     {results['fusion_precision']:.4f}")
            report.append(f"DS Fusion Recall:        {results['fusion_recall']:.4f}")
            report.append(f"DS Fusion F1 Score:      {results['fusion_f1']:.4f}")
            report.append(f"Simple Voting Accuracy:  {results['voting_accuracy']:.4f}")

            # 性能提升分析
            report.append("\n[Performance Improvement Analysis]")
            report.append("─" * 100)

            acc_improvement = results['fusion_accuracy'] - results['avg_individual_acc']
            acc_improvement_pct = (acc_improvement / results['avg_individual_acc']) * 100

            prec_improvement = results['fusion_precision'] - np.mean(results['individual_precisions'])
            prec_improvement_pct = (prec_improvement / np.mean(results['individual_precisions'])) * 100

            rec_improvement = results['fusion_recall'] - np.mean(results['individual_recalls'])
            rec_improvement_pct = (rec_improvement / np.mean(results['individual_recalls'])) * 100

            f1_improvement = results['fusion_f1'] - np.mean(results['individual_f1s'])
            f1_improvement_pct = (f1_improvement / np.mean(results['individual_f1s'])) * 100

            report.append(f"Accuracy Improvement:    {acc_improvement:+.4f} ({acc_improvement_pct:+.2f}%)")
            report.append(f"Precision Improvement:   {prec_improvement:+.4f} ({prec_improvement_pct:+.2f}%)")
            report.append(f"Recall Improvement:      {rec_improvement:+.4f} ({rec_improvement_pct:+.2f}%)")
            report.append(f"F1 Score Improvement:    {f1_improvement:+.4f} ({f1_improvement_pct:+.2f}%)")

            # 与最佳单模型对比
            best_improvement = results['fusion_accuracy'] - results['best_individual_acc']
            best_improvement_pct = (best_improvement / results['best_individual_acc']) * 100

            report.append(f"\nComparison with Best Individual Model:")
            report.append(f"  Improvement: {best_improvement:+.4f} ({best_improvement_pct:+.2f}%)")

            # 与简单投票对比
            voting_improvement = results['fusion_accuracy'] - results['voting_accuracy']
            voting_improvement_pct = (voting_improvement / results['voting_accuracy']) * 100

            report.append(f"\nComparison with Simple Voting:")
            report.append(f"  Improvement: {voting_improvement:+.4f} ({voting_improvement_pct:+.2f}%)")

            # 置信度统计
            report.append("\n[Confidence Statistics]")
            report.append("─" * 100)
            confidence = results['confidence']
            report.append(f"Mean Confidence:         {np.mean(confidence):.4f}")
            report.append(f"Median Confidence:       {np.median(confidence):.4f}")
            report.append(f"Std Confidence:          {np.std(confidence):.4f}")
            report.append(f"Min Confidence:          {np.min(confidence):.4f}")
            report.append(f"Max Confidence:          {np.max(confidence):.4f}")

            # 结论
            report.append("\n[Conclusion]")
            report.append("─" * 100)
            if acc_improvement > 0:
                report.append(f"✓ DS fusion IMPROVES performance by {acc_improvement_pct:.2f}%")
            else:
                report.append(f"✗ DS fusion shows {acc_improvement_pct:.2f}% change (consider ensemble diversity)")

            if results['fusion_accuracy'] > results['best_individual_acc']:
                report.append(f"✓ DS fusion OUTPERFORMS the best individual model")
            else:
                report.append(f"  DS fusion performs similarly to the best individual model")

            if results['fusion_accuracy'] > results['voting_accuracy']:
                report.append(f"✓ DS fusion OUTPERFORMS simple voting by {voting_improvement_pct:.2f}%")
            else:
                report.append(f"  Simple voting performs comparably to DS fusion")

    # 总体总结
    report.append("\n\n" + "=" * 100)
    report.append("OVERALL SUMMARY")
    report.append("=" * 100)

    # 计算所有数据集和模型类型的平均提升
    all_improvements = []
    for dataset in ['iris', 'cancer']:
        for model_type in ['nn', 'svm']:
            results = fusion_results[dataset][model_type]
            improvement = results['fusion_accuracy'] - results['avg_individual_acc']
            improvement_pct = (improvement / results['avg_individual_acc']) * 100
            all_improvements.append(improvement_pct)

    avg_improvement = np.mean(all_improvements)

    report.append(f"\nAverage Performance Improvement Across All Experiments: {avg_improvement:.2f}%")
    report.append(f"\nDS Evidence Theory Fusion demonstrates {'POSITIVE' if avg_improvement > 0 else 'MIXED'} results")
    report.append("in combining multiple classifiers for improved prediction accuracy.")

    report.append("\n" + "=" * 100)
    report.append("REPORT GENERATION COMPLETE")
    report.append("=" * 100)

    # 保存报告
    report_text = '\n'.join(report)
    with open('figures/ds_fusion_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 打印报告
    print(report_text)

    return report_text


generate_fusion_report()

# ===================== 10. 最终总结 =====================
print("\n" + "=" * 100)
print("DS证据理论融合任务完成总结")
print("=" * 100)
print(f"✓ 已对 2 个数据集的 40 个模型执行DS证据理论融合")
print(f"✓ 已生成 6 张高质量可视化图表:")
print(f"    1. ds_fusion_comparison.png - 融合效果综合对比")
print(f"    2. ds_fusion_detailed_metrics.png - 详细性能指标对比")
print(f"    3. ds_fusion_confusion_matrices.png - 混淆矩阵")
print(f"    4. ds_fusion_confidence_analysis.png - 置信度分析")
print(f"    5. ds_fusion_improvement_radar.png - 性能提升雷达图")
print(f"✓ 已生成详细分析报告: ds_fusion_report.txt")
print(f"\n所有结果保存在 'figures/' 文件夹中")
print("=" * 100)

# 打印关键发现
print("\n" + "=" * 100)
print("关键发现 (KEY FINDINGS)")
print("=" * 100)

for dataset in ['iris', 'cancer']:
    print(f"\n{dataset.upper()} 数据集:")
    for model_type in ['nn', 'svm']:
        results = fusion_results[dataset][model_type]
        improvement = ((results['fusion_accuracy'] - results['avg_individual_acc']) /
                       results['avg_individual_acc']) * 100
        print(f"  {model_type.upper()}: DS融合准确率 {results['fusion_accuracy']:.4f}, "
              f"提升 {improvement:+.2f}%")

print("\n" + "=" * 100)
print("任务全部完成！")
print("=" * 100)
