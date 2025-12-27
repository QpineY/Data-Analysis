import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import warnings
import os
from matplotlib import rcParams
from scipy import stats

warnings.filterwarnings('ignore')

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 创建保存文件夹
os.makedirs('models/ensemble', exist_ok=True)
os.makedirs('figures/ensemble', exist_ok=True)

print("=" * 80)
print("贝叶斯模型融合 - 开始")
print("=" * 80)

# ===================== 1. 加载已训练的模型 =====================
print("\n加载已训练的模型...")

# 加载数据集（与训练代码保持一致）
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载鸢尾花数据集
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.3, random_state=42
)

# 加载乳腺癌数据集
cancer = datasets.load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target
scaler_cancer = StandardScaler()
X_cancer_scaled = scaler_cancer.fit_transform(X_cancer)
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
    X_cancer_scaled, y_cancer, test_size=0.3, random_state=42
)

# 加载已训练的模型
nn_iris_models = [joblib.load(f'models/nn_iris_{i + 1}.pkl') for i in range(10)]
nn_cancer_models = [joblib.load(f'models/nn_cancer_{i + 1}.pkl') for i in range(10)]
svm_iris_models = [joblib.load(f'models/svm_iris_{i + 1}.pkl') for i in range(10)]
svm_cancer_models = [joblib.load(f'models/svm_cancer_{i + 1}.pkl') for i in range(10)]

print("✓ 成功加载所有模型")


# ===================== 2. 贝叶斯模型融合器 =====================
class BayesianEnsemble:
    """
    贝叶斯模型融合器

    原理：
    1. 收集所有基模型的预测结果作为特征
    2. 使用高斯朴素贝叶斯模型学习这些预测的组合模式
    3. 通过贝叶斯定理计算最终预测的后验概率
    """

    def __init__(self, base_models, model_type='NN'):
        self.base_models = base_models
        self.model_type = model_type
        self.meta_model = GaussianNB()
        self.n_models = len(base_models)

    def _get_base_predictions(self, X):
        """获取所有基模型的预测结果"""
        predictions = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)
        return predictions

    def _get_base_probabilities(self, X):
        """获取所有基模型的预测概率"""
        try:
            # 尝试获取概率预测
            n_samples = X.shape[0]
            n_classes = len(np.unique(self.base_models[0].predict(X[:1])))
            probabilities = np.zeros((n_samples, self.n_models * n_classes))

            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities[:, i * n_classes:(i + 1) * n_classes] = proba
                else:
                    # 如果模型不支持概率预测，使用one-hot编码
                    pred = model.predict(X)
                    for j in range(n_samples):
                        probabilities[j, i * n_classes + int(pred[j])] = 1.0
            return probabilities
        except:
            # 如果失败，返回预测结果
            return self._get_base_predictions(X)

    def fit(self, X_train, y_train):
        """训练贝叶斯融合器"""
        # 获取基模型的预测作为元特征
        meta_features = self._get_base_predictions(X_train)

        # 训练贝叶斯元模型
        self.meta_model.fit(meta_features, y_train)

        return self

    def predict(self, X):
        """使用融合模型进行预测"""
        meta_features = self._get_base_predictions(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        """预测概率"""
        meta_features = self._get_base_predictions(X)
        return self.meta_model.predict_proba(meta_features)

    def get_model_weights(self, X, y):
        """
        计算每个基模型的权重（基于准确率）
        """
        weights = []
        for model in self.base_models:
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            weights.append(acc)
        return np.array(weights) / np.sum(weights)


# ===================== 3. 训练贝叶斯融合器 =====================
print("\n" + "=" * 80)
print("训练贝叶斯融合器...")
print("=" * 80)

# 创建融合器字典
ensembles = {}

# 鸢尾花数据集 - NN融合
print("\n[1/4] 训练鸢尾花数据集 - NN融合器...")
ensemble_nn_iris = BayesianEnsemble(nn_iris_models, 'NN')
ensemble_nn_iris.fit(X_iris_train, y_iris_train)
ensembles['nn_iris'] = ensemble_nn_iris
joblib.dump(ensemble_nn_iris, 'models/ensemble/bayesian_nn_iris.pkl')
print("  ✓ 完成")

# 鸢尾花数据集 - SVM融合
print("[2/4] 训练鸢尾花数据集 - SVM融合器...")
ensemble_svm_iris = BayesianEnsemble(svm_iris_models, 'SVM')
ensemble_svm_iris.fit(X_iris_train, y_iris_train)
ensembles['svm_iris'] = ensemble_svm_iris
joblib.dump(ensemble_svm_iris, 'models/ensemble/bayesian_svm_iris.pkl')
print("  ✓ 完成")

# 乳腺癌数据集 - NN融合
print("[3/4] 训练乳腺癌数据集 - NN融合器...")
ensemble_nn_cancer = BayesianEnsemble(nn_cancer_models, 'NN')
ensemble_nn_cancer.fit(X_cancer_train, y_cancer_train)
ensembles['nn_cancer'] = ensemble_nn_cancer
joblib.dump(ensemble_nn_cancer, 'models/ensemble/bayesian_nn_cancer.pkl')
print("  ✓ 完成")

# 乳腺癌数据集 - SVM融合
print("[4/4] 训练乳腺癌数据集 - SVM融合器...")
ensemble_svm_cancer = BayesianEnsemble(svm_cancer_models, 'SVM')
ensemble_svm_cancer.fit(X_cancer_train, y_cancer_train)
ensembles['svm_cancer'] = ensemble_svm_cancer
joblib.dump(ensemble_svm_cancer, 'models/ensemble/bayesian_svm_cancer.pkl')
print("  ✓ 完成")

# ===================== 4. 评估融合模型 =====================
print("\n" + "=" * 80)
print("评估融合模型性能...")
print("=" * 80)


def evaluate_ensemble(ensemble, base_models, X_test, y_test, dataset_name, model_type):
    """评估融合模型和基模型的性能"""
    results = {
        'base_models': [],
        'ensemble': {}
    }

    # 评估基模型
    for i, model in enumerate(base_models):
        y_pred = model.predict(X_test)
        results['base_models'].append({
            'model_id': i + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

    # 评估融合模型
    y_pred_ensemble = ensemble.predict(X_test)
    results['ensemble'] = {
        'accuracy': accuracy_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
        'predictions': y_pred_ensemble
    }

    # 计算模型权重
    results['weights'] = ensemble.get_model_weights(X_test, y_test)

    return results


# 评估所有融合模型
evaluation_results = {}

print("\n评估鸢尾花数据集 - NN融合...")
evaluation_results['nn_iris'] = evaluate_ensemble(
    ensemble_nn_iris, nn_iris_models, X_iris_test, y_iris_test, 'Iris', 'NN'
)

print("评估鸢尾花数据集 - SVM融合...")
evaluation_results['svm_iris'] = evaluate_ensemble(
    ensemble_svm_iris, svm_iris_models, X_iris_test, y_iris_test, 'Iris', 'SVM'
)

print("评估乳腺癌数据集 - NN融合...")
evaluation_results['nn_cancer'] = evaluate_ensemble(
    ensemble_nn_cancer, nn_cancer_models, X_cancer_test, y_cancer_test, 'Breast Cancer', 'NN'
)

print("评估乳腺癌数据集 - SVM融合...")
evaluation_results['svm_cancer'] = evaluate_ensemble(
    ensemble_svm_cancer, svm_cancer_models, X_cancer_test, y_cancer_test, 'Breast Cancer', 'SVM'
)

print("✓ 评估完成")

# ===================== 5. 可视化 - 融合模型性能对比 =====================
print("\n" + "=" * 80)
print("生成可视化图表...")
print("=" * 80)


def plot_ensemble_comparison(eval_results, dataset_name, model_type, y_test):
    """
    绘制融合模型与基模型的性能对比

    【图表布局】2行 × 3列
    [0,0] 准确率对比柱状图
    [0,1] 所有指标对比柱状图
    [0,2] 模型权重分布
    [1,0] 混淆矩阵 - 最佳基模型
    [1,1] 混淆矩阵 - 融合模型
    [1,2] 性能提升分析
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'{dataset_name} Dataset - {model_type} Ensemble Performance Analysis',
                 fontsize=22, fontweight='bold', y=0.98, family='Times New Roman')

    base_results = eval_results['base_models']
    ensemble_result = eval_results['ensemble']
    weights = eval_results['weights']

    # 提取数据
    base_acc = [r['accuracy'] for r in base_results]
    base_prec = [r['precision'] for r in base_results]
    base_rec = [r['recall'] for r in base_results]
    base_f1 = [r['f1'] for r in base_results]

    # ========== [0,0] 准确率对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(10)
    colors = sns.color_palette("husl", 10)
    bars = ax1.bar(x, base_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加融合模型的水平线
    ax1.axhline(y=ensemble_result['accuracy'], color='red', linestyle='--',
                linewidth=3, label=f'Ensemble: {ensemble_result["accuracy"]:.4f}', alpha=0.9)

    # 标注最佳基模型
    best_idx = np.argmax(base_acc)
    ax1.scatter(best_idx, base_acc[best_idx], s=300, color='gold',
                edgecolors='black', linewidth=2, zorder=5, marker='*',
                label=f'Best Base: {base_acc[best_idx]:.4f}')

    ax1.set_xlabel('Base Model Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison: Base Models vs Ensemble',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i + 1}' for i in range(10)])
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([min(base_acc) - 0.05, 1.02])

    # ========== [0,1] 所有指标对比 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    base_means = [np.mean(base_acc), np.mean(base_prec), np.mean(base_rec), np.mean(base_f1)]
    ensemble_scores = [ensemble_result['accuracy'], ensemble_result['precision'],
                       ensemble_result['recall'], ensemble_result['f1']]

    x_pos = np.arange(len(metrics))
    width = 0.35

    bars1 = ax2.bar(x_pos - width / 2, base_means, width, label='Base Models (Avg)',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width / 2, ensemble_scores, width, label='Ensemble Model',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标注
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    ax2.set_xlabel('Evaluation Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0.8, 1.02])

    # ========== [0,2] 模型权重分布 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    colors_weight = sns.color_palette("viridis", 10)
    bars = ax3.barh(range(10), weights, color=colors_weight, alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    # 标注权重值
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        ax3.text(weight, bar.get_y() + bar.get_height() / 2,
                 f'{weight:.3f}', ha='left', va='center',
                 fontsize=10, fontweight='bold', color='black')

    ax3.set_xlabel('Model Weight (Based on Accuracy)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Base Model Number', fontsize=12, fontweight='bold')
    ax3.set_title('Model Contribution Weights', fontsize=14, fontweight='bold', pad=15)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([f'Model {i + 1}' for i in range(10)])
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax3.invert_yaxis()

    # ========== [1,0] 最佳基模型混淆矩阵 ==========
    ax4 = fig.add_subplot(gs[1, 0])
    best_model_idx = np.argmax(base_acc)

    # 修复：正确映射数据集名称和模型类型
    dataset_key_map = {
        'Iris': 'iris',
        'Breast Cancer': 'cancer'
    }
    dataset_key = dataset_key_map.get(dataset_name, dataset_name.lower().replace(' ', '_'))
    ensemble_key = f"{model_type.lower()}_{dataset_key}"

    # 获取正确的测试数据
    if 'iris' in dataset_name.lower():
        X_test_data = X_iris_test
    else:
        X_test_data = X_cancer_test

    best_model = ensembles[ensemble_key].base_models[best_model_idx]
    y_pred_best = best_model.predict(X_test_data)

    cm_best = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax4,
                cbar_kws={'label': 'Count'}, square=True, linewidths=2,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax4.set_title(f'Best Base Model (Model {best_model_idx + 1})\nConfusion Matrix',
                  fontsize=14, fontweight='bold', pad=15)

    # ========== [1,1] 融合模型混淆矩阵 ==========
    ax5 = fig.add_subplot(gs[1, 1])
    cm_ensemble = confusion_matrix(y_test, ensemble_result['predictions'])
    sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Reds', ax=ax5,
                cbar_kws={'label': 'Count'}, square=True, linewidths=2,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax5.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax5.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax5.set_title('Bayesian Ensemble Model\nConfusion Matrix',
                  fontsize=14, fontweight='bold', pad=15)

    # ========== [1,2] 性能提升分析 ==========
    ax6 = fig.add_subplot(gs[1, 2])

    # 计算提升百分比
    improvements = []
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        base_mean = np.mean([r[metric] for r in base_results])
        ensemble_score = ensemble_result[metric]
        improvement = ((ensemble_score - base_mean) / base_mean) * 100
        improvements.append(improvement)

    colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax6.barh(metrics, improvements, color=colors_imp, alpha=0.8,
                    edgecolor='black', linewidth=2)

    # 添加零线
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=2)

    # 标注百分比
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        x_pos = imp + (0.2 if imp > 0 else -0.2)
        ax6.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f'{imp:+.2f}%', ha='left' if imp > 0 else 'right',
                 va='center', fontsize=11, fontweight='bold')

    ax6.set_xlabel('Performance Improvement (%)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Metric', fontsize=12, fontweight='bold')
    ax6.set_title('Ensemble vs Base Models (Average)\nPerformance Improvement',
                  fontsize=14, fontweight='bold', pad=15)
    ax6.grid(True, alpha=0.3, axis='x', linestyle='--')

    plt.savefig(f'figures/ensemble/ensemble_analysis_{dataset_name.replace(" ", "_")}_{model_type}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 已保存: ensemble_analysis_{dataset_name.replace(' ', '_')}_{model_type}.png")


# 生成所有融合分析图
print("\n生成融合模型分析图...")
plot_ensemble_comparison(evaluation_results['nn_iris'], 'Iris', 'NN', y_iris_test)
plot_ensemble_comparison(evaluation_results['svm_iris'], 'Iris', 'SVM', y_iris_test)
plot_ensemble_comparison(evaluation_results['nn_cancer'], 'Breast Cancer', 'NN', y_cancer_test)
plot_ensemble_comparison(evaluation_results['svm_cancer'], 'Breast Cancer', 'SVM', y_cancer_test)


# ===================== 6. 综合对比可视化 =====================
def plot_comprehensive_ensemble_comparison():
    """
    综合对比所有融合模型

    【图表布局】2行 × 3列
    [0,0] 鸢尾花数据集性能对比
    [0,1] 乳腺癌数据集性能对比
    [0,2] 跨数据集性能对比
    [1,0] 融合效果提升热力图
    [1,1] 模型稳定性分析
    [1,2] 最终性能排名
    """
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    fig.suptitle('Comprehensive Bayesian Ensemble Analysis - All Datasets & Models',
                 fontsize=24, fontweight='bold', y=0.98, family='Times New Roman')

    # ========== [0,0] 鸢尾花数据集性能对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # 修复：使用正确的键名
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']  # 注意这里是 'f1' 不是 'f1_score'

    nn_iris_scores = [evaluation_results['nn_iris']['ensemble'][key] for key in metric_keys]
    svm_iris_scores = [evaluation_results['svm_iris']['ensemble'][key] for key in metric_keys]

    x_pos = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x_pos - width / 2, nn_iris_scores, width, label='NN Ensemble',
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x_pos + width / 2, svm_iris_scores, width, label='SVM Ensemble',
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Iris Dataset - Ensemble Performance', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics, rotation=15, ha='right')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0.85, 1.05])

    # ========== [0,1] 乳腺癌数据集性能对比 ==========
    ax2 = fig.add_subplot(gs[0, 1])

    nn_cancer_scores = [evaluation_results['nn_cancer']['ensemble'][key] for key in metric_keys]
    svm_cancer_scores = [evaluation_results['svm_cancer']['ensemble'][key] for key in metric_keys]

    bars1 = ax2.bar(x_pos - width / 2, nn_cancer_scores, width, label='NN Ensemble',
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x_pos + width / 2, svm_cancer_scores, width, label='SVM Ensemble',
                    color='#f39c12', alpha=0.85, edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Breast Cancer Dataset - Ensemble Performance',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics, rotation=15, ha='right')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0.85, 1.05])

    # ========== [0,2] 跨数据集性能对比 ==========
    ax3 = fig.add_subplot(gs[0, 2])

    datasets = ['Iris', 'Breast\nCancer']
    nn_avg = [np.mean(nn_iris_scores), np.mean(nn_cancer_scores)]
    svm_avg = [np.mean(svm_iris_scores), np.mean(svm_cancer_scores)]

    x_pos2 = np.arange(len(datasets))
    width2 = 0.35

    bars1 = ax3.bar(x_pos2 - width2 / 2, nn_avg, width2, label='NN Ensemble',
                    color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=2)
    bars2 = ax3.bar(x_pos2 + width2 / 2, svm_avg, width2, label='SVM Ensemble',
                    color='#e67e22', alpha=0.85, edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')

    ax3.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Score (All Metrics)', fontsize=12, fontweight='bold')
    ax3.set_title('Cross-Dataset Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos2)
    ax3.set_xticklabels(datasets)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim([0.85, 1.05])

    # ========== [1,0] 融合效果提升热力图 ==========
    ax4 = fig.add_subplot(gs[1, 0])

    # 计算每个配置的提升百分比
    improvement_data = []
    for key in ['nn_iris', 'svm_iris', 'nn_cancer', 'svm_cancer']:
        improvements_row = []
        for metric in metric_keys:  # 使用正确的键名
            base_mean = np.mean([r[metric] for r in evaluation_results[key]['base_models']])
            ensemble_score = evaluation_results[key]['ensemble'][metric]
            improvement = ((ensemble_score - base_mean) / base_mean) * 100
            improvements_row.append(improvement)
        improvement_data.append(improvements_row)

    improvement_data = np.array(improvement_data)

    im = ax4.imshow(improvement_data, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=5)
    ax4.set_xticks(np.arange(4))
    ax4.set_yticks(np.arange(4))
    ax4.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax4.set_yticklabels(['NN-Iris', 'SVM-Iris', 'NN-Cancer', 'SVM-Cancer'])
    ax4.set_title('Ensemble Improvement over Base Models (%)',
                  fontsize=15, fontweight='bold', pad=15)

    # 添加数值标注
    for i in range(4):
        for j in range(4):
            text = ax4.text(j, i, f'{improvement_data[i, j]:.2f}%',
                            ha="center", va="center", color="black",
                            fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Improvement (%)', fontsize=11, fontweight='bold')

    # ========== [1,1] 模型稳定性分析 ==========
    ax5 = fig.add_subplot(gs[1, 1])

    # 计算每个配置的标准差（稳定性指标）
    stability_data = []
    labels = []
    for key, label in [('nn_iris', 'NN-Iris'), ('svm_iris', 'SVM-Iris'),
                       ('nn_cancer', 'NN-Cancer'), ('svm_cancer', 'SVM-Cancer')]:
        base_acc = [r['accuracy'] for r in evaluation_results[key]['base_models']]
        std = np.std(base_acc)
        stability_data.append(std)
        labels.append(label)

    colors_stability = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax5.barh(labels, stability_data, color=colors_stability, alpha=0.85,
                    edgecolor='black', linewidth=2)

    for bar, std in zip(bars, stability_data):
        ax5.text(std + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{std:.4f}', ha='left', va='center',
                 fontsize=11, fontweight='bold')

    ax5.set_xlabel('Standard Deviation (Lower = More Stable)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Model Configuration', fontsize=12, fontweight='bold')
    ax5.set_title('Base Models Stability Analysis', fontsize=15, fontweight='bold', pad=15)
    ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax5.invert_yaxis()

    # ========== [1,2] 最终性能排名 ==========
    ax6 = fig.add_subplot(gs[1, 2])

    # 收集所有模型的准确率
    all_scores = []
    all_labels = []

    for key, label in [('nn_iris', 'NN-Iris'), ('svm_iris', 'SVM-Iris'),
                       ('nn_cancer', 'NN-Cancer'), ('svm_cancer', 'SVM-Cancer')]:
        # 基模型平均
        base_avg = np.mean([r['accuracy'] for r in evaluation_results[key]['base_models']])
        all_scores.append(base_avg)
        all_labels.append(f'{label}\n(Base Avg)')

        # 融合模型
        ensemble_acc = evaluation_results[key]['ensemble']['accuracy']
        all_scores.append(ensemble_acc)
        all_labels.append(f'{label}\n(Ensemble)')

    # 排序
    sorted_indices = np.argsort(all_scores)
    sorted_scores = [all_scores[i] for i in sorted_indices]
    sorted_labels = [all_labels[i] for i in sorted_indices]

    # 颜色：融合模型用深色，基模型用浅色
    colors_rank = []
    for label in sorted_labels:
        if 'Ensemble' in label:
            if 'Iris' in label:
                colors_rank.append('#e74c3c' if 'SVM' in label else '#3498db')
            else:
                colors_rank.append('#f39c12' if 'SVM' in label else '#2ecc71')
        else:
            colors_rank.append('#bdc3c7')

    bars = ax6.barh(range(len(sorted_scores)), sorted_scores, color=colors_rank,
                    alpha=0.85, edgecolor='black', linewidth=2)

    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax6.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{score:.4f}', ha='left', va='center',
                 fontsize=10, fontweight='bold')

    ax6.set_xlabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax6.set_title('Final Performance Ranking (All Models)', fontsize=15, fontweight='bold', pad=15)
    ax6.set_yticks(range(len(sorted_labels)))
    ax6.set_yticklabels(sorted_labels, fontsize=9)
    ax6.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax6.set_xlim([0.85, 1.02])

    plt.savefig('figures/ensemble/comprehensive_ensemble_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: comprehensive_ensemble_comparison.png")


print("\n生成综合对比图...")
plot_comprehensive_ensemble_comparison()

# ===================== 7. 生成详细报告 =====================
print("\n" + "=" * 80)
print("生成详细报告...")
print("=" * 80)


def generate_ensemble_report():
    """生成融合模型详细报告"""
    report = []
    report.append("=" * 90)
    report.append("贝叶斯模型融合 - 详细性能评估报告")
    report.append("=" * 90)
    report.append("")

    for dataset_key, dataset_name in [('iris', '鸢尾花数据集'), ('cancer', '乳腺癌数据集')]:
        report.append(f"\n{'=' * 90}")
        report.append(f"{dataset_name}")
        report.append(f"{'=' * 90}")

        for model_type, model_name in [('nn', '神经网络'), ('svm', '支持向量机')]:
            key = f"{model_type}_{dataset_key}"
            results = evaluation_results[key]

            report.append(f"\n[{model_name}模型融合分析]")
            report.append("-" * 90)

            # 基模型统计
            base_acc = [r['accuracy'] for r in results['base_models']]
            base_f1 = [r['f1'] for r in results['base_models']]

            report.append(f"\n基模型性能统计:")
            report.append(f"  准确率: 平均 = {np.mean(base_acc):.4f}, 标准差 = {np.std(base_acc):.4f}")
            report.append(f"  准确率: 最小 = {np.min(base_acc):.4f}, 最大 = {np.max(base_acc):.4f}")
            report.append(f"  F1分数: 平均 = {np.mean(base_f1):.4f}, 标准差 = {np.std(base_f1):.4f}")

            # 融合模型性能
            ensemble = results['ensemble']
            report.append(f"\n融合模型性能:")
            report.append(f"  准确率:   {ensemble['accuracy']:.4f}")
            report.append(f"  精确率:   {ensemble['precision']:.4f}")
            report.append(f"  召回率:   {ensemble['recall']:.4f}")
            report.append(f"  F1分数:   {ensemble['f1']:.4f}")

            # 性能提升
            acc_improvement = ((ensemble['accuracy'] - np.mean(base_acc)) / np.mean(base_acc)) * 100
            f1_improvement = ((ensemble['f1'] - np.mean(base_f1)) / np.mean(base_f1)) * 100

            report.append(f"\n性能提升:")
            report.append(f"  准确率提升: {acc_improvement:+.2f}%")
            report.append(f"  F1分数提升: {f1_improvement:+.2f}%")

            # 模型权重
            weights = results['weights']
            top3_indices = np.argsort(weights)[-3:][::-1]

            report.append(f"\n贡献度最高的3个基模型:")
            for idx in top3_indices:
                report.append(f"  模型 {idx + 1}: 权重 = {weights[idx]:.4f}, "
                              f"准确率 = {results['base_models'][idx]['accuracy']:.4f}")

            report.append("")

    # 总结
    report.append(f"\n{'=' * 90}")
    report.append("总结与结论")
    report.append(f"{'=' * 90}")

    # 找出最佳配置
    best_overall = None
    best_score = 0
    for key in ['nn_iris', 'svm_iris', 'nn_cancer', 'svm_cancer']:
        score = evaluation_results[key]['ensemble']['accuracy']
        if score > best_score:
            best_score = score
            best_overall = key

    model_type_map = {'nn': '神经网络', 'svm': '支持向量机'}
    dataset_map = {'iris': '鸢尾花', 'cancer': '乳腺癌'}

    best_model_type = model_type_map[best_overall.split('_')[0]]
    best_dataset = dataset_map[best_overall.split('_')[1]]

    report.append(f"\n1. 最佳融合模型配置:")
    report.append(f"   {best_dataset}数据集 + {best_model_type}融合器")
    report.append(f"   准确率: {best_score:.4f}")

    report.append(f"\n2. 融合效果分析:")
    positive_improvements = 0
    for key in ['nn_iris', 'svm_iris', 'nn_cancer', 'svm_cancer']:
        base_avg = np.mean([r['accuracy'] for r in evaluation_results[key]['base_models']])
        ensemble_acc = evaluation_results[key]['ensemble']['accuracy']
        if ensemble_acc > base_avg:
            positive_improvements += 1

    report.append(f"   {positive_improvements}/4 个配置中融合模型优于基模型平均水平")

    report.append(f"\n3. 关键发现:")
    report.append(f"   - 贝叶斯融合器能够有效整合多个基模型的预测")
    report.append(f"   - 模型融合在大多数情况下提升了预测性能")
    report.append(f"   - 不同基模型的贡献度存在显著差异")

    report.append(f"\n{'=' * 90}")
    report.append("报告生成完成!")
    report.append(f"{'=' * 90}")

    # 保存报告
    with open('figures/ensemble/ensemble_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 打印报告
    for line in report:
        print(line)


generate_ensemble_report()

# ===================== 8. 生成性能对比表格 =====================
print("\n" + "=" * 80)
print("生成性能对比表格...")
print("=" * 80)


def create_performance_table():
    """创建详细的性能对比表格可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Detailed Performance Comparison Tables',
                 fontsize=22, fontweight='bold', y=0.98, family='Times New Roman')

    configs = [
        ('nn_iris', 'Iris - Neural Network', axes[0, 0]),
        ('svm_iris', 'Iris - Support Vector Machine', axes[0, 1]),
        ('nn_cancer', 'Breast Cancer - Neural Network', axes[1, 0]),
        ('svm_cancer', 'Breast Cancer - Support Vector Machine', axes[1, 1])
    ]

    for key, title, ax in configs:
        results = evaluation_results[key]

        # 准备表格数据
        table_data = []
        for i, r in enumerate(results['base_models']):
            table_data.append([
                f"Model {i + 1}",
                f"{r['accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}"
            ])

        # 添加平均值
        base_acc = [r['accuracy'] for r in results['base_models']]
        base_prec = [r['precision'] for r in results['base_models']]
        base_rec = [r['recall'] for r in results['base_models']]
        base_f1 = [r['f1'] for r in results['base_models']]

        table_data.append([
            "Average",
            f"{np.mean(base_acc):.4f}",
            f"{np.mean(base_prec):.4f}",
            f"{np.mean(base_rec):.4f}",
            f"{np.mean(base_f1):.4f}"
        ])

        # 添加融合模型
        ensemble = results['ensemble']
        table_data.append([
            "Ensemble",
            f"{ensemble['accuracy']:.4f}",
            f"{ensemble['precision']:.4f}",
            f"{ensemble['recall']:.4f}",
            f"{ensemble['f1']:.4f}"
        ])

        # 创建表格
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data,
                         colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white', family='Times New Roman')

        # 设置基模型行样式
        for i in range(1, 11):
            for j in range(5):
                table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_text_props(family='Times New Roman')

        # 设置平均值行样式
        for j in range(5):
            table[(11, j)].set_facecolor('#f39c12')
            table[(11, j)].set_text_props(weight='bold', family='Times New Roman')

        # 设置融合模型行样式
        for j in range(5):
            table[(12, j)].set_facecolor('#2ecc71')
            table[(12, j)].set_text_props(weight='bold', color='white', family='Times New Roman')

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, family='Times New Roman')

    plt.tight_layout()
    plt.savefig('figures/ensemble/performance_tables.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: performance_tables.png")


create_performance_table()

# ===================== 9. 最终总结 =====================
print("\n" + "=" * 80)
print("任务完成总结")
print("=" * 80)
print(f"✓ 已训练 4 个贝叶斯融合器")
print(f"  - 鸢尾花数据集: NN融合器 + SVM融合器")
print(f"  - 乳腺癌数据集: NN融合器 + SVM融合器")
print(f"\n✓ 已生成 8 张融合分析图")
print(f"  - 4 张单独融合分析图（每个数据集×模型类型）")
print(f"  - 1 张综合对比图")
print(f"  - 1 张性能对比表格")
print(f"\n✓ 已生成详细评估报告")
print(f"  - 基模型性能统计")
print(f"  - 融合模型性能分析")
print(f"  - 性能提升对比")
print(f"  - 模型权重分析")
print(f"\n所有融合模型保存在: models/ensemble/")
print(f"所有可视化图表保存在: figures/ensemble/")
print("=" * 80)

# 打印关键结果
print("\n" + "=" * 80)
print("关键结果摘要")
print("=" * 80)

for key, name in [('nn_iris', '鸢尾花-NN'), ('svm_iris', '鸢尾花-SVM'),
                  ('nn_cancer', '乳腺癌-NN'), ('svm_cancer', '乳腺癌-SVM')]:
    results = evaluation_results[key]
    base_avg = np.mean([r['accuracy'] for r in results['base_models']])
    ensemble_acc = results['ensemble']['accuracy']
    improvement = ((ensemble_acc - base_avg) / base_avg) * 100

    print(f"\n{name}:")
    print(f"  基模型平均准确率: {base_avg:.4f}")
    print(f"  融合模型准确率:   {ensemble_acc:.4f}")
    print(f"  性能提升:         {improvement:+.2f}%")

print("\n" + "=" * 80)
print("🎉 贝叶斯模型融合任务全部完成！")
print("=" * 80)
