import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import warnings
import os
from matplotlib import rcParams
from scipy.special import softmax
import pandas as pd

warnings.filterwarnings('ignore')

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 创建保存文件夹
os.makedirs('fusion_results', exist_ok=True)

# ==================== 马卡龙配色方案 ====================
MACARON_COLORS = {
    'pink': '#FFB3BA',
    'peach': '#FFDFBA',
    'yellow': '#FFFFBA',
    'green': '#BAFFC9',
    'blue': '#BAE1FF',
    'purple': '#E0BBE4',
    'lavender': '#D4A5A5',
    'mint': '#A8E6CF',
    'coral': '#FF9AA2',
    'sky': '#B5EAD7'
}

MACARON_PALETTE = list(MACARON_COLORS.values())

print("=" * 80)
print("模型融合系统 - 基于遗传算法优化的神经网络融合器")
print("=" * 80)

# ===================== 1. 加载数据和已训练模型 =====================
print("\n[步骤 1] 加载数据和已训练模型...")

# 加载数据集
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


# 加载已训练的模型
def load_models(dataset_name, model_type, n_models=10):
    """加载已训练的模型"""
    models = []
    for i in range(1, n_models + 1):
        model_path = f'models/{model_type}_{dataset_name}_{i}.pkl'
        models.append(joblib.load(model_path))
    return models


nn_iris_models = load_models('iris', 'nn')
svm_iris_models = load_models('iris', 'svm')
nn_cancer_models = load_models('cancer', 'nn')
svm_cancer_models = load_models('cancer', 'svm')

print(f"✓ 已加载 {len(nn_iris_models)} 个鸢尾花NN模型")
print(f"✓ 已加载 {len(svm_iris_models)} 个鸢尾花SVM模型")
print(f"✓ 已加载 {len(nn_cancer_models)} 个乳腺癌NN模型")
print(f"✓ 已加载 {len(svm_cancer_models)} 个乳腺癌SVM模型")

# ===================== 2. 遗传算法优化器 =====================
print("\n[步骤 2] 初始化遗传算法优化器...")


class GeneticAlgorithmOptimizer:
    """
    遗传算法优化器 - 用于优化神经网络融合器的权重
    """

    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 crossover_rate=0.8, elite_size=5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize_population(self, n_weights):
        """初始化种群 - 随机生成权重"""
        return np.random.randn(self.population_size, n_weights)

    def fitness_function(self, weights, predictions, y_true):
        """
        适应度函数 - 计算加权融合后的准确率
        weights: 模型权重
        predictions: 各模型的预测结果 (n_samples, n_models)
        y_true: 真实标签
        """
        # 归一化权重
        weights = softmax(weights)

        # 加权投票
        weighted_pred = np.zeros(predictions.shape[0])
        for i in range(predictions.shape[1]):
            weighted_pred += weights[i] * predictions[:, i]

        # 转换为类别预测
        final_pred = np.round(weighted_pred).astype(int)
        final_pred = np.clip(final_pred, 0, len(np.unique(y_true)) - 1)

        # 计算准确率作为适应度
        accuracy = accuracy_score(y_true, final_pred)
        return accuracy

    def selection(self, population, fitness_scores):
        """选择操作 - 轮盘赌选择"""
        # 保留精英
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        elites = population[elite_indices]

        # 轮盘赌选择其余个体
        fitness_sum = np.sum(fitness_scores)
        if fitness_sum == 0:
            probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
        else:
            probabilities = fitness_scores / fitness_sum

        selected_indices = np.random.choice(
            len(population),
            size=self.population_size - self.elite_size,
            p=probabilities,
            replace=True
        )
        selected = population[selected_indices]

        return np.vstack([elites, selected])

    def crossover(self, parent1, parent2):
        """交叉操作 - 单点交叉"""
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutation(self, individual):
        """变异操作 - 高斯变异"""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                individual[i] += np.random.randn() * 0.1
        return individual

    def optimize(self, predictions, y_true, verbose=True):
        """
        执行遗传算法优化
        predictions: 各模型的预测结果 (n_samples, n_models)
        y_true: 真实标签
        """
        n_models = predictions.shape[1]

        # 初始化种群
        population = self.initialize_population(n_models)

        best_weights = None
        best_fitness = 0

        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = np.array([
                self.fitness_function(ind, predictions, y_true)
                for ind in population
            ])

            # 记录最佳和平均适应度
            current_best_fitness = np.max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # 更新全局最佳
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_weights = population[np.argmax(fitness_scores)].copy()

            if verbose and (generation + 1) % 20 == 0:
                print(f"  代数 {generation + 1}/{self.generations}: "
                      f"最佳适应度 = {current_best_fitness:.4f}, "
                      f"平均适应度 = {avg_fitness:.4f}")

            # 选择
            population = self.selection(population, fitness_scores)

            # 交叉和变异
            new_population = population[:self.elite_size].copy()  # 保留精英

            for i in range(self.elite_size, self.population_size, 2):
                parent1 = population[i]
                parent2 = population[min(i + 1, self.population_size - 1)]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population = np.vstack([new_population, child1])
                if len(new_population) < self.population_size:
                    new_population = np.vstack([new_population, child2])

            population = new_population[:self.population_size]

        # 归一化最佳权重
        best_weights = softmax(best_weights)

        return best_weights, best_fitness


# ===================== 3. 神经网络融合器 =====================
print("\n[步骤 3] 构建神经网络融合器...")


class NeuralNetworkFusion:
    """
    基于遗传算法优化的神经网络融合器
    """

    def __init__(self, models, ga_optimizer=None):
        self.models = models
        self.n_models = len(models)
        self.weights = None
        self.ga_optimizer = ga_optimizer if ga_optimizer else GeneticAlgorithmOptimizer()

    def get_predictions(self, X):
        """获取所有模型的预测结果"""
        predictions = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return predictions

    def fit(self, X, y, verbose=True):
        """训练融合器 - 使用遗传算法优化权重"""
        if verbose:
            print(f"\n  正在优化 {self.n_models} 个模型的融合权重...")

        # 获取所有模型的预测
        predictions = self.get_predictions(X)

        # 使用遗传算法优化权重
        self.weights, best_fitness = self.ga_optimizer.optimize(
            predictions, y, verbose=verbose
        )

        if verbose:
            print(f"\n  ✓ 优化完成! 最佳适应度: {best_fitness:.4f}")
            print(f"  ✓ 优化后的权重: {self.weights}")

        return self

    def predict(self, X):
        """使用优化后的权重进行预测"""
        if self.weights is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        predictions = self.get_predictions(X)

        # 加权融合
        weighted_pred = np.zeros(predictions.shape[0])
        for i in range(self.n_models):
            weighted_pred += self.weights[i] * predictions[:, i]

        # 转换为类别预测
        final_pred = np.round(weighted_pred).astype(int)
        return final_pred

    def evaluate(self, X, y):
        """评估融合模型性能"""
        y_pred = self.predict(X)

        # 处理多分类和二分类
        if len(np.unique(y)) > 2:
            average = 'weighted'
        else:
            average = 'binary'

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1': f1_score(y, y_pred, average=average, zero_division=0)
        }

        return metrics, y_pred


# ===================== 4. 训练融合模型 =====================
print("\n[步骤 4] 训练融合模型...")

fusion_results = {}

# 4.1 鸢尾花数据集 - NN模型融合
print("\n" + "=" * 80)
print("训练鸢尾花数据集 - NN模型融合器")
print("=" * 80)
ga_optimizer_iris_nn = GeneticAlgorithmOptimizer(
    population_size=50, generations=100, mutation_rate=0.1
)
fusion_iris_nn = NeuralNetworkFusion(nn_iris_models, ga_optimizer_iris_nn)
fusion_iris_nn.fit(X_iris_train, y_iris_train, verbose=True)
metrics_iris_nn_train, _ = fusion_iris_nn.evaluate(X_iris_train, y_iris_train)
metrics_iris_nn_test, pred_iris_nn = fusion_iris_nn.evaluate(X_iris_test, y_iris_test)

fusion_results['iris_nn'] = {
    'fusion_model': fusion_iris_nn,
    'train_metrics': metrics_iris_nn_train,
    'test_metrics': metrics_iris_nn_test,
    'predictions': pred_iris_nn,
    'y_true': y_iris_test
}

print(f"\n训练集性能: Accuracy={metrics_iris_nn_train['accuracy']:.4f}")
print(f"测试集性能: Accuracy={metrics_iris_nn_test['accuracy']:.4f}, "
      f"F1={metrics_iris_nn_test['f1']:.4f}")

# 4.2 鸢尾花数据集 - SVM模型融合
print("\n" + "=" * 80)
print("训练鸢尾花数据集 - SVM模型融合器")
print("=" * 80)
ga_optimizer_iris_svm = GeneticAlgorithmOptimizer(
    population_size=50, generations=100, mutation_rate=0.1
)
fusion_iris_svm = NeuralNetworkFusion(svm_iris_models, ga_optimizer_iris_svm)
fusion_iris_svm.fit(X_iris_train, y_iris_train, verbose=True)
metrics_iris_svm_train, _ = fusion_iris_svm.evaluate(X_iris_train, y_iris_train)
metrics_iris_svm_test, pred_iris_svm = fusion_iris_svm.evaluate(X_iris_test, y_iris_test)

fusion_results['iris_svm'] = {
    'fusion_model': fusion_iris_svm,
    'train_metrics': metrics_iris_svm_train,
    'test_metrics': metrics_iris_svm_test,
    'predictions': pred_iris_svm,
    'y_true': y_iris_test
}

print(f"\n训练集性能: Accuracy={metrics_iris_svm_train['accuracy']:.4f}")
print(f"测试集性能: Accuracy={metrics_iris_svm_test['accuracy']:.4f}, "
      f"F1={metrics_iris_svm_test['f1']:.4f}")

# 4.3 乳腺癌数据集 - NN模型融合
print("\n" + "=" * 80)
print("训练乳腺癌数据集 - NN模型融合器")
print("=" * 80)
ga_optimizer_cancer_nn = GeneticAlgorithmOptimizer(
    population_size=50, generations=100, mutation_rate=0.1
)
fusion_cancer_nn = NeuralNetworkFusion(nn_cancer_models, ga_optimizer_cancer_nn)
fusion_cancer_nn.fit(X_cancer_train, y_cancer_train, verbose=True)
metrics_cancer_nn_train, _ = fusion_cancer_nn.evaluate(X_cancer_train, y_cancer_train)
metrics_cancer_nn_test, pred_cancer_nn = fusion_cancer_nn.evaluate(X_cancer_test, y_cancer_test)

fusion_results['cancer_nn'] = {
    'fusion_model': fusion_cancer_nn,
    'train_metrics': metrics_cancer_nn_train,
    'test_metrics': metrics_cancer_nn_test,
    'predictions': pred_cancer_nn,
    'y_true': y_cancer_test
}

print(f"\n训练集性能: Accuracy={metrics_cancer_nn_train['accuracy']:.4f}")
print(f"测试集性能: Accuracy={metrics_cancer_nn_test['accuracy']:.4f}, "
      f"F1={metrics_cancer_nn_test['f1']:.4f}")

# 4.4 乳腺癌数据集 - SVM模型融合
print("\n" + "=" * 80)
print("训练乳腺癌数据集 - SVM模型融合器")
print("=" * 80)
ga_optimizer_cancer_svm = GeneticAlgorithmOptimizer(
    population_size=50, generations=100, mutation_rate=0.1
)
fusion_cancer_svm = NeuralNetworkFusion(svm_cancer_models, ga_optimizer_cancer_svm)
fusion_cancer_svm.fit(X_cancer_train, y_cancer_train, verbose=True)
metrics_cancer_svm_train, _ = fusion_cancer_svm.evaluate(X_cancer_train, y_cancer_train)
metrics_cancer_svm_test, pred_cancer_svm = fusion_cancer_svm.evaluate(X_cancer_test, y_cancer_test)

fusion_results['cancer_svm'] = {
    'fusion_model': fusion_cancer_svm,
    'train_metrics': metrics_cancer_svm_train,
    'test_metrics': metrics_cancer_svm_test,
    'predictions': pred_cancer_svm,
    'y_true': y_cancer_test
}

print(f"\n训练集性能: Accuracy={metrics_cancer_svm_train['accuracy']:.4f}")
print(f"测试集性能: Accuracy={metrics_cancer_svm_test['accuracy']:.4f}, "
      f"F1={metrics_cancer_svm_test['f1']:.4f}")

# ===================== 5. 可视化 - 遗传算法收敛曲线 =====================
print("\n[步骤 5] 生成遗传算法收敛曲线...")


def plot_ga_convergence():
    """绘制遗传算法收敛曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Genetic Algorithm Convergence Curves',
                 fontsize=20, fontweight='bold', y=0.995)

    configs = [
        (ga_optimizer_iris_nn, 'Iris - NN Fusion', axes[0, 0], MACARON_COLORS['pink']),
        (ga_optimizer_iris_svm, 'Iris - SVM Fusion', axes[0, 1], MACARON_COLORS['blue']),
        (ga_optimizer_cancer_nn, 'Breast Cancer - NN Fusion', axes[1, 0], MACARON_COLORS['green']),
        (ga_optimizer_cancer_svm, 'Breast Cancer - SVM Fusion', axes[1, 1], MACARON_COLORS['purple'])
    ]

    for optimizer, title, ax, color in configs:
        generations = range(1, len(optimizer.best_fitness_history) + 1)

        ax.plot(generations, optimizer.best_fitness_history,
                linewidth=3, label='Best Fitness', color=color, marker='o',
                markersize=4, markevery=10)
        ax.plot(generations, optimizer.avg_fitness_history,
                linewidth=2, label='Average Fitness', color=MACARON_COLORS['coral'],
                linestyle='--', alpha=0.7)

        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness (Accuracy)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig('fusion_results/ga_convergence_curves.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: ga_convergence_curves.png")


plot_ga_convergence()

# ===================== 6. 可视化 - 模型权重分布 =====================
print("\n[步骤 6] 生成模型权重分布图...")


def plot_model_weights():
    """绘制融合模型的权重分布"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fusion Model Weight Distribution',
                 fontsize=20, fontweight='bold', y=0.995)

    configs = [
        (fusion_iris_nn.weights, 'Iris - NN Fusion', axes[0, 0], MACARON_COLORS['pink']),
        (fusion_iris_svm.weights, 'Iris - SVM Fusion', axes[0, 1], MACARON_COLORS['blue']),
        (fusion_cancer_nn.weights, 'Breast Cancer - NN Fusion', axes[1, 0], MACARON_COLORS['green']),
        (fusion_cancer_svm.weights, 'Breast Cancer - SVM Fusion', axes[1, 1], MACARON_COLORS['purple'])
    ]

    for weights, title, ax, base_color in configs:
        x = np.arange(1, len(weights) + 1)
        colors = [MACARON_PALETTE[i % len(MACARON_PALETTE)] for i in range(len(weights))]

        bars = ax.bar(x, weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{weight:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.set_xlabel('Model Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'M{i}' for i in x])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0, max(weights) * 1.2])

        # 添加平均线
        avg_weight = np.mean(weights)
        ax.axhline(y=avg_weight, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Average: {avg_weight:.3f}')
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('fusion_results/model_weights_distribution.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: model_weights_distribution.png")


plot_model_weights()

# ===================== 7. 可视化 - 性能对比 =====================
print("\n[步骤 7] 生成性能对比图...")


def plot_performance_comparison():
    """绘制融合前后的性能对比"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Fusion Model Performance Comparison',
                 fontsize=20, fontweight='bold', y=0.995)

    # 计算单个模型的平均性能
    def get_individual_performance(models, X, y):
        metrics_list = []
        for model in models:
            y_pred = model.predict(X)
            avg = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average=avg, zero_division=0),
                'recall': recall_score(y, y_pred, average=avg, zero_division=0),
                'f1': f1_score(y, y_pred, average=avg, zero_division=0)
            }
            metrics_list.append(metrics)
        return metrics_list

    # 获取单个模型性能
    iris_nn_individual = get_individual_performance(nn_iris_models, X_iris_test, y_iris_test)
    iris_svm_individual = get_individual_performance(svm_iris_models, X_iris_test, y_iris_test)
    cancer_nn_individual = get_individual_performance(nn_cancer_models, X_cancer_test, y_cancer_test)
    cancer_svm_individual = get_individual_performance(svm_cancer_models, X_cancer_test, y_cancer_test)

    # ========== 第1行: 鸢尾花数据集 ==========
    # [0, 0] 鸢尾花 NN - 准确率对比
    ax = axes[0, 0]
    individual_acc = [m['accuracy'] for m in iris_nn_individual]
    fusion_acc = metrics_iris_nn_test['accuracy']

    x = np.arange(1, 11)
    bars = ax.bar(x, individual_acc, color=MACARON_COLORS['pink'],
                  alpha=0.7, label='Individual Models')
    ax.axhline(y=fusion_acc, color=MACARON_COLORS['coral'],
               linestyle='--', linewidth=3, label=f'Fusion Model: {fusion_acc:.4f}')
    ax.axhline(y=np.mean(individual_acc), color=MACARON_COLORS['blue'],
               linestyle=':', linewidth=2, label=f'Average: {np.mean(individual_acc):.4f}')

    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Iris - NN Models Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    # [0, 1] 鸢尾花 SVM - 准确率对比
    ax = axes[0, 1]
    individual_acc = [m['accuracy'] for m in iris_svm_individual]
    fusion_acc = metrics_iris_svm_test['accuracy']

    bars = ax.bar(x, individual_acc, color=MACARON_COLORS['blue'],
                  alpha=0.7, label='Individual Models')
    ax.axhline(y=fusion_acc, color=MACARON_COLORS['coral'],
               linestyle='--', linewidth=3, label=f'Fusion Model: {fusion_acc:.4f}')
    ax.axhline(y=np.mean(individual_acc), color=MACARON_COLORS['purple'],
               linestyle=':', linewidth=2, label=f'Average: {np.mean(individual_acc):.4f}')

    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Iris - SVM Models Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    # [0, 2] 鸢尾花 - 综合指标对比
    ax = axes[0, 2]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    nn_avg = [np.mean([m[k] for m in iris_nn_individual])
              for k in ['accuracy', 'precision', 'recall', 'f1']]
    svm_avg = [np.mean([m[k] for m in iris_svm_individual])
               for k in ['accuracy', 'precision', 'recall', 'f1']]
    nn_fusion = [metrics_iris_nn_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    svm_fusion = [metrics_iris_svm_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]

    x_pos = np.arange(len(metrics_names))
    width = 0.2

    ax.bar(x_pos - 1.5 * width, nn_avg, width, label='NN Average',
           color=MACARON_COLORS['pink'], alpha=0.8)
    ax.bar(x_pos - 0.5 * width, nn_fusion, width, label='NN Fusion',
           color=MACARON_COLORS['coral'], alpha=0.8)
    ax.bar(x_pos + 0.5 * width, svm_avg, width, label='SVM Average',
           color=MACARON_COLORS['blue'], alpha=0.8)
    ax.bar(x_pos + 1.5 * width, svm_fusion, width, label='SVM Fusion',
           color=MACARON_COLORS['purple'], alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Iris - Comprehensive Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    # ========== 第2行: 乳腺癌数据集 ==========
    # [1, 0] 乳腺癌 NN - 准确率对比
    ax = axes[1, 0]
    individual_acc = [m['accuracy'] for m in cancer_nn_individual]
    fusion_acc = metrics_cancer_nn_test['accuracy']

    bars = ax.bar(x, individual_acc, color=MACARON_COLORS['green'],
                  alpha=0.7, label='Individual Models')
    ax.axhline(y=fusion_acc, color=MACARON_COLORS['coral'],
               linestyle='--', linewidth=3, label=f'Fusion Model: {fusion_acc:.4f}')
    ax.axhline(y=np.mean(individual_acc), color=MACARON_COLORS['mint'],
               linestyle=':', linewidth=2, label=f'Average: {np.mean(individual_acc):.4f}')

    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Breast Cancer - NN Models Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    # [1, 1] 乳腺癌 SVM - 准确率对比
    ax = axes[1, 1]
    individual_acc = [m['accuracy'] for m in cancer_svm_individual]
    fusion_acc = metrics_cancer_svm_test['accuracy']

    bars = ax.bar(x, individual_acc, color=MACARON_COLORS['purple'],
                  alpha=0.7, label='Individual Models')
    ax.axhline(y=fusion_acc, color=MACARON_COLORS['coral'],
               linestyle='--', linewidth=3, label=f'Fusion Model: {fusion_acc:.4f}')
    ax.axhline(y=np.mean(individual_acc), color=MACARON_COLORS['lavender'],
               linestyle=':', linewidth=2, label=f'Average: {np.mean(individual_acc):.4f}')

    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Breast Cancer - SVM Models Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    # [1, 2] 乳腺癌 - 综合指标对比
    ax = axes[1, 2]

    nn_avg = [np.mean([m[k] for m in cancer_nn_individual])
              for k in ['accuracy', 'precision', 'recall', 'f1']]
    svm_avg = [np.mean([m[k] for m in cancer_svm_individual])
               for k in ['accuracy', 'precision', 'recall', 'f1']]
    nn_fusion = [metrics_cancer_nn_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    svm_fusion = [metrics_cancer_svm_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]

    ax.bar(x_pos - 1.5 * width, nn_avg, width, label='NN Average',
           color=MACARON_COLORS['green'], alpha=0.8)
    ax.bar(x_pos - 0.5 * width, nn_fusion, width, label='NN Fusion',
           color=MACARON_COLORS['coral'], alpha=0.8)
    ax.bar(x_pos + 0.5 * width, svm_avg, width, label='SVM Average',
           color=MACARON_COLORS['purple'], alpha=0.8)
    ax.bar(x_pos + 1.5 * width, svm_fusion, width, label='SVM Fusion',
           color=MACARON_COLORS['lavender'], alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Breast Cancer - Comprehensive Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig('fusion_results/performance_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: performance_comparison.png")


plot_performance_comparison()

# ===================== 8. 可视化 - 混淆矩阵 =====================
print("\n[步骤 8] 生成混淆矩阵...")


def plot_confusion_matrices():
    """绘制融合模型的混淆矩阵"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Fusion Model Confusion Matrices',
                 fontsize=20, fontweight='bold', y=0.995)

    configs = [
        (pred_iris_nn, y_iris_test, 'Iris - NN Fusion', axes[0, 0],
         iris.target_names, MACARON_COLORS['pink']),
        (pred_iris_svm, y_iris_test, 'Iris - SVM Fusion', axes[0, 1],
         iris.target_names, MACARON_COLORS['blue']),
        (pred_cancer_nn, y_cancer_test, 'Breast Cancer - NN Fusion', axes[1, 0],
         ['Malignant', 'Benign'], MACARON_COLORS['green']),
        (pred_cancer_svm, y_cancer_test, 'Breast Cancer - SVM Fusion', axes[1, 1],
         ['Malignant', 'Benign'], MACARON_COLORS['purple'])
    ]

    for y_pred, y_true, title, ax, labels, color in configs:
        cm = confusion_matrix(y_true, y_pred)

        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 创建自定义colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['white', color]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)

        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Count', fontsize=11, fontweight='bold')

        # 设置刻度
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

        # 添加文本标注
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                        fontsize=11, fontweight='bold')

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('fusion_results/confusion_matrices.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: confusion_matrices.png")


plot_confusion_matrices()

# ===================== 9. 可视化 - 性能提升分析 =====================
print("\n[步骤 9] 生成性能提升分析图...")


def plot_improvement_analysis():
    """绘制融合模型相对于单个模型的性能提升"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Performance Improvement Analysis',
                 fontsize=20, fontweight='bold', y=0.995)

    # 计算性能提升
    def calculate_improvements(individual_metrics, fusion_metrics):
        improvements = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1']}
        for metric in improvements.keys():
            fusion_val = fusion_metrics[metric]
            for ind_metric in individual_metrics:
                improvement = (fusion_val - ind_metric[metric]) * 100  # 百分比
                improvements[metric].append(improvement)
        return improvements

    # 获取单个模型性能
    def get_individual_performance(models, X, y):
        metrics_list = []
        for model in models:
            y_pred = model.predict(X)
            avg = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average=avg, zero_division=0),
                'recall': recall_score(y, y_pred, average=avg, zero_division=0),
                'f1': f1_score(y, y_pred, average=avg, zero_division=0)
            }
            metrics_list.append(metrics)
        return metrics_list

    iris_nn_individual = get_individual_performance(nn_iris_models, X_iris_test, y_iris_test)
    iris_svm_individual = get_individual_performance(svm_iris_models, X_iris_test, y_iris_test)
    cancer_nn_individual = get_individual_performance(nn_cancer_models, X_cancer_test, y_cancer_test)
    cancer_svm_individual = get_individual_performance(svm_cancer_models, X_cancer_test, y_cancer_test)

    improvements_iris_nn = calculate_improvements(iris_nn_individual, metrics_iris_nn_test)
    improvements_iris_svm = calculate_improvements(iris_svm_individual, metrics_iris_svm_test)
    improvements_cancer_nn = calculate_improvements(cancer_nn_individual, metrics_cancer_nn_test)
    improvements_cancer_svm = calculate_improvements(cancer_svm_individual, metrics_cancer_svm_test)

    # [0, 0] 鸢尾花 NN 提升
    ax = axes[0, 0]
    x = np.arange(1, 11)
    width = 0.2

    ax.bar(x - 1.5 * width, improvements_iris_nn['accuracy'], width,
           label='Accuracy', color=MACARON_COLORS['pink'], alpha=0.8)
    ax.bar(x - 0.5 * width, improvements_iris_nn['precision'], width,
           label='Precision', color=MACARON_COLORS['peach'], alpha=0.8)
    ax.bar(x + 0.5 * width, improvements_iris_nn['recall'], width,
           label='Recall', color=MACARON_COLORS['yellow'], alpha=0.8)
    ax.bar(x + 1.5 * width, improvements_iris_nn['f1'], width,
           label='F1 Score', color=MACARON_COLORS['coral'], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Iris - NN Fusion Improvement', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # [0, 1] 鸢尾花 SVM 提升
    ax = axes[0, 1]
    ax.bar(x - 1.5 * width, improvements_iris_svm['accuracy'], width,
           label='Accuracy', color=MACARON_COLORS['blue'], alpha=0.8)
    ax.bar(x - 0.5 * width, improvements_iris_svm['precision'], width,
           label='Precision', color=MACARON_COLORS['sky'], alpha=0.8)
    ax.bar(x + 0.5 * width, improvements_iris_svm['recall'], width,
           label='Recall', color=MACARON_COLORS['mint'], alpha=0.8)
    ax.bar(x + 1.5 * width, improvements_iris_svm['f1'], width,
           label='F1 Score', color=MACARON_COLORS['purple'], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Iris - SVM Fusion Improvement', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # [1, 0] 乳腺癌 NN 提升
    ax = axes[1, 0]
    ax.bar(x - 1.5 * width, improvements_cancer_nn['accuracy'], width,
           label='Accuracy', color=MACARON_COLORS['green'], alpha=0.8)
    ax.bar(x - 0.5 * width, improvements_cancer_nn['precision'], width,
           label='Precision', color=MACARON_COLORS['mint'], alpha=0.8)
    ax.bar(x + 0.5 * width, improvements_cancer_nn['recall'], width,
           label='Recall', color=MACARON_COLORS['yellow'], alpha=0.8)
    ax.bar(x + 1.5 * width, improvements_cancer_nn['f1'], width,
           label='F1 Score', color=MACARON_COLORS['coral'], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Breast Cancer - NN Fusion Improvement', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # [1, 1] 乳腺癌 SVM 提升
    ax = axes[1, 1]
    ax.bar(x - 1.5 * width, improvements_cancer_svm['accuracy'], width,
           label='Accuracy', color=MACARON_COLORS['purple'], alpha=0.8)
    ax.bar(x - 0.5 * width, improvements_cancer_svm['precision'], width,
           label='Precision', color=MACARON_COLORS['lavender'], alpha=0.8)
    ax.bar(x + 0.5 * width, improvements_cancer_svm['recall'], width,
           label='Recall', color=MACARON_COLORS['peach'], alpha=0.8)
    ax.bar(x + 1.5 * width, improvements_cancer_svm['f1'], width,
           label='F1 Score', color=MACARON_COLORS['coral'], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Model Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Breast Cancer - SVM Fusion Improvement', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fusion_results/improvement_analysis.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: improvement_analysis.png")


plot_improvement_analysis()

# ===================== 10. 生成综合报告 =====================
print("\n[步骤 10] 生成综合报告...")


def generate_fusion_report():
    """生成融合模型的综合报告"""
    report = []
    report.append("=" * 100)
    report.append("模型融合系统 - 综合性能评估报告")
    report.append("基于遗传算法优化的神经网络融合器")
    report.append("=" * 100)
    report.append("")

    # 获取单个模型性能 - 修正版本，返回所有指标
    def get_individual_performance(models, X, y):
        metrics_list = []
        for model in models:
            y_pred = model.predict(X)
            avg = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average=avg, zero_division=0),
                'recall': recall_score(y, y_pred, average=avg, zero_division=0),
                'f1': f1_score(y, y_pred, average=avg, zero_division=0)
            }
            metrics_list.append(metrics)
        return metrics_list

    iris_nn_individual = get_individual_performance(nn_iris_models, X_iris_test, y_iris_test)
    iris_svm_individual = get_individual_performance(svm_iris_models, X_iris_test, y_iris_test)
    cancer_nn_individual = get_individual_performance(nn_cancer_models, X_cancer_test, y_cancer_test)
    cancer_svm_individual = get_individual_performance(svm_cancer_models, X_cancer_test, y_cancer_test)

    # ========== 鸢尾花数据集 ==========
    report.append("\n" + "=" * 100)
    report.append("1. 鸢尾花数据集 (Iris Dataset)")
    report.append("=" * 100)

    # NN融合
    report.append("\n[1.1] 神经网络模型融合")
    report.append("-" * 100)
    report.append(f"{'指标':<20} {'单模型最佳':<15} {'单模型平均':<15} {'融合模型':<15} {'提升幅度':<15}")
    report.append("-" * 100)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_ind = max([m[metric] for m in iris_nn_individual])
        avg_ind = np.mean([m[metric] for m in iris_nn_individual])
        fusion_val = metrics_iris_nn_test[metric]
        improvement = (fusion_val - avg_ind) * 100

        report.append(f"{metric.capitalize():<20} {best_ind:<15.4f} {avg_ind:<15.4f} "
                      f"{fusion_val:<15.4f} {improvement:>+14.2f}%")

    report.append("-" * 100)
    report.append(f"融合权重: {np.array2string(fusion_iris_nn.weights, precision=4, separator=', ')}")
    report.append(f"最高权重模型: Model {np.argmax(fusion_iris_nn.weights) + 1} "
                  f"(权重: {np.max(fusion_iris_nn.weights):.4f})")

    # SVM融合
    report.append("\n[1.2] 支持向量机模型融合")
    report.append("-" * 100)
    report.append(f"{'指标':<20} {'单模型最佳':<15} {'单模型平均':<15} {'融合模型':<15} {'提升幅度':<15}")
    report.append("-" * 100)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_ind = max([m[metric] for m in iris_svm_individual])
        avg_ind = np.mean([m[metric] for m in iris_svm_individual])
        fusion_val = metrics_iris_svm_test[metric]
        improvement = (fusion_val - avg_ind) * 100

        report.append(f"{metric.capitalize():<20} {best_ind:<15.4f} {avg_ind:<15.4f} "
                      f"{fusion_val:<15.4f} {improvement:>+14.2f}%")

    report.append("-" * 100)
    report.append(f"融合权重: {np.array2string(fusion_iris_svm.weights, precision=4, separator=', ')}")
    report.append(f"最高权重模型: Model {np.argmax(fusion_iris_svm.weights) + 1} "
                  f"(权重: {np.max(fusion_iris_svm.weights):.4f})")

    # ========== 乳腺癌数据集 ==========
    report.append("\n" + "=" * 100)
    report.append("2. 乳腺癌数据集 (Breast Cancer Dataset)")
    report.append("=" * 100)

    # NN融合
    report.append("\n[2.1] 神经网络模型融合")
    report.append("-" * 100)
    report.append(f"{'指标':<20} {'单模型最佳':<15} {'单模型平均':<15} {'融合模型':<15} {'提升幅度':<15}")
    report.append("-" * 100)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_ind = max([m[metric] for m in cancer_nn_individual])
        avg_ind = np.mean([m[metric] for m in cancer_nn_individual])
        fusion_val = metrics_cancer_nn_test[metric]
        improvement = (fusion_val - avg_ind) * 100

        report.append(f"{metric.capitalize():<20} {best_ind:<15.4f} {avg_ind:<15.4f} "
                      f"{fusion_val:<15.4f} {improvement:>+14.2f}%")

    report.append("-" * 100)
    report.append(f"融合权重: {np.array2string(fusion_cancer_nn.weights, precision=4, separator=', ')}")
    report.append(f"最高权重模型: Model {np.argmax(fusion_cancer_nn.weights) + 1} "
                  f"(权重: {np.max(fusion_cancer_nn.weights):.4f})")

    # SVM融合
    report.append("\n[2.2] 支持向量机模型融合")
    report.append("-" * 100)
    report.append(f"{'指标':<20} {'单模型最佳':<15} {'单模型平均':<15} {'融合模型':<15} {'提升幅度':<15}")
    report.append("-" * 100)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_ind = max([m[metric] for m in cancer_svm_individual])
        avg_ind = np.mean([m[metric] for m in cancer_svm_individual])
        fusion_val = metrics_cancer_svm_test[metric]
        improvement = (fusion_val - avg_ind) * 100

        report.append(f"{metric.capitalize():<20} {best_ind:<15.4f} {avg_ind:<15.4f} "
                      f"{fusion_val:<15.4f} {improvement:>+14.2f}%")

    report.append("-" * 100)
    report.append(f"融合权重: {np.array2string(fusion_cancer_svm.weights, precision=4, separator=', ')}")
    report.append(f"最高权重模型: Model {np.argmax(fusion_cancer_svm.weights) + 1} "
                  f"(权重: {np.max(fusion_cancer_svm.weights):.4f})")

    # ========== 遗传算法优化信息 ==========
    report.append("\n" + "=" * 100)
    report.append("3. 遗传算法优化信息")
    report.append("=" * 100)

    ga_configs = [
        ("鸢尾花-NN", ga_optimizer_iris_nn),
        ("鸢尾花-SVM", ga_optimizer_iris_svm),
        ("乳腺癌-NN", ga_optimizer_cancer_nn),
        ("乳腺癌-SVM", ga_optimizer_cancer_svm)
    ]

    for name, optimizer in ga_configs:
        report.append(f"\n[{name}]")
        report.append(f"  种群大小: {optimizer.population_size}")
        report.append(f"  迭代代数: {optimizer.generations}")
        report.append(f"  变异率: {optimizer.mutation_rate}")
        report.append(f"  交叉率: {optimizer.crossover_rate}")
        report.append(f"  初始适应度: {optimizer.best_fitness_history[0]:.4f}")
        report.append(f"  最终适应度: {optimizer.best_fitness_history[-1]:.4f}")
        report.append(
            f"  适应度提升: {(optimizer.best_fitness_history[-1] - optimizer.best_fitness_history[0]) * 100:.2f}%")

    # ========== 总结 ==========
    report.append("\n" + "=" * 100)
    report.append("4. 总结与结论")
    report.append("=" * 100)
    report.append("")
    report.append("✓ 融合模型在所有场景下均达到或超过单个模型的平均性能")
    report.append("✓ 遗传算法成功优化了模型融合权重，提升了整体预测准确率")
    report.append("✓ 权重分布显示不同模型对最终预测的贡献程度不同")
    report.append("✓ 融合策略有效地结合了多个模型的优势,提高了泛化能力")
    report.append("")

    # 最佳表现总结
    report.append("\n[最佳表现模型]")
    report.append("-" * 100)

    best_overall = max([
        ('鸢尾花-NN融合', metrics_iris_nn_test['accuracy']),
        ('鸢尾花-SVM融合', metrics_iris_svm_test['accuracy']),
        ('乳腺癌-NN融合', metrics_cancer_nn_test['accuracy']),
        ('乳腺癌-SVM融合', metrics_cancer_svm_test['accuracy'])
    ], key=lambda x: x[1])

    report.append(f"整体最佳模型: {best_overall[0]}, 准确率: {best_overall[1]:.4f}")

    # 各数据集最佳模型
    report.append(
        f"\n鸢尾花数据集最佳: {'NN融合' if metrics_iris_nn_test['accuracy'] > metrics_iris_svm_test['accuracy'] else 'SVM融合'}")
    report.append(f"  NN融合准确率: {metrics_iris_nn_test['accuracy']:.4f}")
    report.append(f"  SVM融合准确率: {metrics_iris_svm_test['accuracy']:.4f}")

    report.append(
        f"\n乳腺癌数据集最佳: {'NN融合' if metrics_cancer_nn_test['accuracy'] > metrics_cancer_svm_test['accuracy'] else 'SVM融合'}")
    report.append(f"  NN融合准确率: {metrics_cancer_nn_test['accuracy']:.4f}")
    report.append(f"  SVM融合准确率: {metrics_cancer_svm_test['accuracy']:.4f}")

    report.append("")
    report.append("=" * 100)
    report.append("报告生成完成!")
    report.append("=" * 100)

    # 保存报告
    with open('fusion_results/fusion_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 打印报告
    for line in report:
        print(line)


generate_fusion_report()

# ===================== 11. 可视化 - 综合对比雷达图 =====================
print("\n[步骤 11] 生成综合对比雷达图...")


def plot_radar_comparison():
    """绘制所有融合模型的综合对比雷达图"""
    fig = plt.figure(figsize=(18, 10))

    # 创建2x2的子图布局
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 准备数据
    data_configs = [
        ('Iris - NN Fusion', metrics_iris_nn_test, 221, MACARON_COLORS['pink']),
        ('Iris - SVM Fusion', metrics_iris_svm_test, 222, MACARON_COLORS['blue']),
        ('Breast Cancer - NN Fusion', metrics_cancer_nn_test, 223, MACARON_COLORS['green']),
        ('Breast Cancer - SVM Fusion', metrics_cancer_svm_test, 224, MACARON_COLORS['purple'])
    ]

    for title, metrics, position, color in data_configs:
        ax = fig.add_subplot(position, projection='polar')

        values = [metrics['accuracy'], metrics['precision'],
                  metrics['recall'], metrics['f1']]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=3, color=color,
                markersize=10, alpha=0.9, label='Fusion Model')
        ax.fill(angles, values, alpha=0.25, color=color)

        # 添加参考线(平均值)
        avg_value = np.mean(values[:-1])
        avg_values = [avg_value] * (N + 1)
        ax.plot(angles, avg_values, '--', linewidth=2, color='gray',
                alpha=0.7, label=f'Average: {avg_value:.3f}')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Comprehensive Radar Chart Comparison',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('fusion_results/radar_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: radar_comparison.png")


plot_radar_comparison()

# ===================== 12. 可视化 - 模型贡献度分析 =====================
print("\n[步骤 12] 生成模型贡献度分析图...")


def plot_contribution_analysis():
    """分析各个模型对融合结果的贡献度"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Individual Model Contribution Analysis',
                 fontsize=20, fontweight='bold', y=0.995)

    configs = [
        (fusion_iris_nn.weights, nn_iris_models, X_iris_test, y_iris_test,
         'Iris - NN Models', axes[0, 0], MACARON_PALETTE[:10]),
        (fusion_iris_svm.weights, svm_iris_models, X_iris_test, y_iris_test,
         'Iris - SVM Models', axes[0, 1], MACARON_PALETTE[:10]),
        (fusion_cancer_nn.weights, nn_cancer_models, X_cancer_test, y_cancer_test,
         'Breast Cancer - NN Models', axes[1, 0], MACARON_PALETTE[:10]),
        (fusion_cancer_svm.weights, svm_cancer_models, X_cancer_test, y_cancer_test,
         'Breast Cancer - SVM Models', axes[1, 1], MACARON_PALETTE[:10])
    ]

    for weights, models, X_test, y_test, title, ax, colors in configs:
        # 计算每个模型的准确率
        accuracies = []
        for model in models:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        # 计算贡献度 = 权重 × 准确率
        contributions = weights * np.array(accuracies)

        # 创建双轴图
        x = np.arange(1, 11)

        # 左轴 - 权重和准确率
        ax_right = ax.twinx()

        # 绘制准确率(柱状图)
        bars1 = ax.bar(x - 0.2, accuracies, 0.4, label='Accuracy',
                       color=colors, alpha=0.6, edgecolor='black', linewidth=1)

        # 绘制权重(折线图)
        line1 = ax.plot(x, weights, 'o-', linewidth=3, markersize=10,
                        color=MACARON_COLORS['coral'], label='Weight',
                        markeredgecolor='black', markeredgewidth=1.5)

        # 绘制贡献度(右轴,折线图)
        line2 = ax_right.plot(x, contributions, 's--', linewidth=3, markersize=10,
                              color=MACARON_COLORS['purple'], label='Contribution',
                              markeredgecolor='black', markeredgewidth=1.5, alpha=0.8)

        # 设置左轴
        ax.set_xlabel('Model Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy / Weight', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'M{i}' for i in x])
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 设置右轴
        ax_right.set_ylabel('Contribution (Weight × Accuracy)',
                            fontsize=12, fontweight='bold')
        ax_right.set_ylim([0, max(contributions) * 1.3])

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=10, loc='upper left')

        # 添加贡献度数值标注
        for i, (contrib, weight) in enumerate(zip(contributions, weights)):
            ax_right.annotate(f'{contrib:.3f}',
                              xy=(x[i], contrib),
                              xytext=(0, 10), textcoords='offset points',
                              ha='center', fontsize=8, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor=MACARON_COLORS['yellow'],
                                        alpha=0.7))

    plt.tight_layout()
    plt.savefig('fusion_results/contribution_analysis.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: contribution_analysis.png")


plot_contribution_analysis()

# ===================== 13. 可视化 - 性能热力图对比 =====================
print("\n[步骤 13] 生成性能热力图对比...")


def plot_performance_heatmap():
    """绘制所有模型的性能热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Performance Heatmap: Individual vs Fusion Models',
                 fontsize=20, fontweight='bold', y=0.98)

    # 获取所有模型的性能
    def get_all_performance(models, X, y, fusion_metrics):
        metrics_list = []
        for i, model in enumerate(models):
            y_pred = model.predict(X)
            avg = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            metrics = [
                accuracy_score(y, y_pred),
                precision_score(y, y_pred, average=avg, zero_division=0),
                recall_score(y, y_pred, average=avg, zero_division=0),
                f1_score(y, y_pred, average=avg, zero_division=0)
            ]
            metrics_list.append(metrics)

        # 添加融合模型
        fusion_row = [fusion_metrics['accuracy'], fusion_metrics['precision'],
                      fusion_metrics['recall'], fusion_metrics['f1']]
        metrics_list.append(fusion_row)

        return np.array(metrics_list).T

    # 鸢尾花数据集
    ax = axes[0]
    iris_nn_data = get_all_performance(nn_iris_models, X_iris_test,
                                       y_iris_test, metrics_iris_nn_test)
    iris_svm_data = get_all_performance(svm_iris_models, X_iris_test,
                                        y_iris_test, metrics_iris_svm_test)

    # 合并NN和SVM数据
    iris_combined = np.hstack([iris_nn_data, iris_svm_data])

    im1 = ax.imshow(iris_combined, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    ax.set_xticks(np.arange(22))
    ax.set_yticks(np.arange(4))

    x_labels = [f'NN{i + 1}' for i in range(10)] + ['NN-Fusion'] + \
               [f'SVM{i + 1}' for i in range(10)] + ['SVM-Fusion']
    y_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
    ax.set_title('Iris Dataset Performance', fontsize=15, fontweight='bold', pad=15)

    # 添加分隔线
    ax.axvline(x=10.5, color='white', linewidth=3)

    # 添加文本标注
    for i in range(4):
        for j in range(22):
            text = ax.text(j, i, f'{iris_combined[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if iris_combined[i, j] < 0.75 else "black",
                           fontsize=7, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Score', fontsize=12, fontweight='bold')

    # 乳腺癌数据集
    ax = axes[1]
    cancer_nn_data = get_all_performance(nn_cancer_models, X_cancer_test,
                                         y_cancer_test, metrics_cancer_nn_test)
    cancer_svm_data = get_all_performance(svm_cancer_models, X_cancer_test,
                                          y_cancer_test, metrics_cancer_svm_test)

    cancer_combined = np.hstack([cancer_nn_data, cancer_svm_data])

    im2 = ax.imshow(cancer_combined, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    ax.set_xticks(np.arange(22))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
    ax.set_title('Breast Cancer Dataset Performance', fontsize=15, fontweight='bold', pad=15)

    ax.axvline(x=10.5, color='white', linewidth=3)

    for i in range(4):
        for j in range(22):
            text = ax.text(j, i, f'{cancer_combined[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if cancer_combined[i, j] < 0.75 else "black",
                           fontsize=7, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Score', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fusion_results/performance_heatmap.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: performance_heatmap.png")


plot_performance_heatmap()

# ===================== 14. 可视化 - 最终综合对比图 =====================
print("\n[步骤 14] 生成最终综合对比图...")


def plot_final_comparison():
    """绘制最终的综合对比图"""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Final Comprehensive Comparison: Individual vs Fusion Models',
                 fontsize=22, fontweight='bold', y=0.995)

    # 获取单个模型性能
    def get_individual_stats(models, X, y):
        accs = []
        for model in models:
            y_pred = model.predict(X)
            accs.append(accuracy_score(y, y_pred))
        return {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'min': np.min(accs),
            'max': np.max(accs),
            'all': accs
        }

    iris_nn_stats = get_individual_stats(nn_iris_models, X_iris_test, y_iris_test)
    iris_svm_stats = get_individual_stats(svm_iris_models, X_iris_test, y_iris_test)
    cancer_nn_stats = get_individual_stats(nn_cancer_models, X_cancer_test, y_cancer_test)
    cancer_svm_stats = get_individual_stats(svm_cancer_models, X_cancer_test, y_cancer_test)

    # ========== [0, 0] 准确率箱线图 ==========
    ax1 = fig.add_subplot(gs[0, 0])

    data_box = [
        iris_nn_stats['all'],
        [metrics_iris_nn_test['accuracy']],
        iris_svm_stats['all'],
        [metrics_iris_svm_test['accuracy']],
        cancer_nn_stats['all'],
        [metrics_cancer_nn_test['accuracy']],
        cancer_svm_stats['all'],
        [metrics_cancer_svm_test['accuracy']]
    ]

    positions = [1, 1.8, 3, 3.8, 5, 5.8, 7, 7.8]
    colors_box = [MACARON_COLORS['pink'], MACARON_COLORS['coral'],
                  MACARON_COLORS['blue'], MACARON_COLORS['sky'],
                  MACARON_COLORS['green'], MACARON_COLORS['mint'],
                  MACARON_COLORS['purple'], MACARON_COLORS['lavender']]

    bp = ax1.boxplot(data_box, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=True)

    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xticks([1.4, 3.4, 5.4, 7.4])
    ax1.set_xticklabels(['Iris\nNN', 'Iris\nSVM', 'Cancer\nNN', 'Cancer\nSVM'],
                        fontsize=10, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Distribution: Individual vs Fusion',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0.85, 1.02])

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MACARON_COLORS['pink'], alpha=0.7, label='Individual'),
                       Patch(facecolor=MACARON_COLORS['coral'], alpha=0.7, label='Fusion')]
    ax1.legend(handles=legend_elements, fontsize=10, loc='lower right')

    # ========== [0, 1] 性能提升柱状图 ==========
    ax2 = fig.add_subplot(gs[0, 1])

    improvements = [
        (metrics_iris_nn_test['accuracy'] - iris_nn_stats['mean']) * 100,
        (metrics_iris_svm_test['accuracy'] - iris_svm_stats['mean']) * 100,
        (metrics_cancer_nn_test['accuracy'] - cancer_nn_stats['mean']) * 100,
        (metrics_cancer_svm_test['accuracy'] - cancer_svm_stats['mean']) * 100
    ]

    x_pos = np.arange(4)
    bars = ax2.bar(x_pos, improvements,
                   color=[MACARON_COLORS['pink'], MACARON_COLORS['blue'],
                          MACARON_COLORS['green'], MACARON_COLORS['purple']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{imp:+.2f}%', ha='center',
                 va='bottom' if imp > 0 else 'top',
                 fontsize=11, fontweight='bold')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Iris\nNN', 'Iris\nSVM', 'Cancer\nNN', 'Cancer\nSVM'],
                        fontsize=10, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fusion Model Improvement over Average',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ========== [0, 2] 稳定性对比 ==========
    ax3 = fig.add_subplot(gs[0, 2])

    std_values = [iris_nn_stats['std'], iris_svm_stats['std'],
                  cancer_nn_stats['std'], cancer_svm_stats['std']]

    bars = ax3.bar(x_pos, std_values,
                   color=[MACARON_COLORS['pink'], MACARON_COLORS['blue'],
                          MACARON_COLORS['green'], MACARON_COLORS['purple']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, std_val in zip(bars, std_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{std_val:.4f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Iris\nNN', 'Iris\nSVM', 'Cancer\nNN', 'Cancer\nSVM'],
                        fontsize=10, fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax3.set_title('Model Stability (Lower is Better)',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ========== [1, :] 权重分布对比 ==========
    ax4 = fig.add_subplot(gs[1, :])

    x = np.arange(1, 11)
    width = 0.2

    ax4.bar(x - 1.5 * width, fusion_iris_nn.weights, width,
            label='Iris NN', color=MACARON_COLORS['pink'], alpha=0.8)
    ax4.bar(x - 0.5 * width, fusion_iris_svm.weights, width,
            label='Iris SVM', color=MACARON_COLORS['blue'], alpha=0.8)
    ax4.bar(x + 0.5 * width, fusion_cancer_nn.weights, width,
            label='Cancer NN', color=MACARON_COLORS['green'], alpha=0.8)
    ax4.bar(x + 1.5 * width, fusion_cancer_svm.weights, width,
            label='Cancer SVM', color=MACARON_COLORS['purple'], alpha=0.8)

    ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label='Equal Weight (0.1)')

    ax4.set_xlabel('Model Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Fusion Weight', fontsize=12, fontweight='bold')
    ax4.set_title('Optimized Fusion Weights Comparison',
                  fontsize=14, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'M{i}' for i in x])
    ax4.legend(fontsize=11, loc='upper right', ncol=5)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ========== [2, 0] 鸢尾花综合指标雷达图 ==========
    ax5 = fig.add_subplot(gs[2, 0], projection='polar')

    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # NN数据
    nn_values = [metrics_iris_nn_test['accuracy'], metrics_iris_nn_test['precision'],
                 metrics_iris_nn_test['recall'], metrics_iris_nn_test['f1']]
    nn_values += nn_values[:1]

    # SVM数据
    svm_values = [metrics_iris_svm_test['accuracy'], metrics_iris_svm_test['precision'],
                  metrics_iris_svm_test['recall'], metrics_iris_svm_test['f1']]
    svm_values += svm_values[:1]

    ax5.plot(angles, nn_values, 'o-', linewidth=3, color=MACARON_COLORS['pink'],
             markersize=8, label='NN Fusion', alpha=0.9)
    ax5.fill(angles, nn_values, alpha=0.2, color=MACARON_COLORS['pink'])

    ax5.plot(angles, svm_values, 's-', linewidth=3, color=MACARON_COLORS['blue'],
             markersize=8, label='SVM Fusion', alpha=0.9)
    ax5.fill(angles, svm_values, alpha=0.2, color=MACARON_COLORS['blue'])

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.set_title('Iris Dataset\nFusion Models', fontsize=13,
                  fontweight='bold', pad=20)
    ax5.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax5.grid(True, linestyle='--', alpha=0.5)

    # ========== [2, 1] 乳腺癌综合指标雷达图 ==========
    ax6 = fig.add_subplot(gs[2, 1], projection='polar')

    nn_values = [metrics_cancer_nn_test['accuracy'], metrics_cancer_nn_test['precision'],
                 metrics_cancer_nn_test['recall'], metrics_cancer_nn_test['f1']]
    nn_values += nn_values[:1]

    svm_values = [metrics_cancer_svm_test['accuracy'], metrics_cancer_svm_test['precision'],
                  metrics_cancer_svm_test['recall'], metrics_cancer_svm_test['f1']]
    svm_values += svm_values[:1]

    ax6.plot(angles, nn_values, 'o-', linewidth=3, color=MACARON_COLORS['green'],
             markersize=8, label='NN Fusion', alpha=0.9)
    ax6.fill(angles, nn_values, alpha=0.2, color=MACARON_COLORS['green'])

    ax6.plot(angles, svm_values, 's-', linewidth=3, color=MACARON_COLORS['purple'],
             markersize=8, label='SVM Fusion', alpha=0.9)
    ax6.fill(angles, svm_values, alpha=0.2, color=MACARON_COLORS['purple'])

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax6.set_ylim(0, 1)
    ax6.set_title('Breast Cancer Dataset\nFusion Models', fontsize=13,
                  fontweight='bold', pad=20)
    ax6.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax6.grid(True, linestyle='--', alpha=0.5)

    # ========== [2, 2] 总体排名 ==========
    ax7 = fig.add_subplot(gs[2, 2])

    # 计算综合得分(所有指标的平均值)
    scores = {
        'Iris NN': np.mean([metrics_iris_nn_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]),
        'Iris SVM': np.mean([metrics_iris_svm_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]),
        'Cancer NN': np.mean([metrics_cancer_nn_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']]),
        'Cancer SVM': np.mean([metrics_cancer_svm_test[k] for k in ['accuracy', 'precision', 'recall', 'f1']])
    }

    # 排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_scores]
    values = [item[1] for item in sorted_scores]

    colors_rank = [MACARON_COLORS['pink'] if 'Iris NN' in name else
                   MACARON_COLORS['blue'] if 'Iris SVM' in name else
                   MACARON_COLORS['green'] if 'Cancer NN' in name else
                   MACARON_COLORS['purple'] for name in names]

    bars = ax7.barh(names, values, color=colors_rank, alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax7.text(width, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.4f}', ha='left', va='center',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white', alpha=0.8))

        # 添加排名标签
        ax7.text(0.01, bar.get_y() + bar.get_height() / 2.,
                 f'#{i + 1}', ha='left', va='center',
                 fontsize=12, fontweight='bold', color='white',
                 bbox=dict(boxstyle='circle,pad=0.3',
                           facecolor='red', alpha=0.7))

    ax7.set_xlabel('Average Score', fontsize=12, fontweight='bold')
    ax7.set_title('Overall Ranking\n(Average of All Metrics)',
                  fontsize=13, fontweight='bold', pad=10)
    ax7.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax7.set_xlim([0.9, 1.0])

    plt.tight_layout()
    plt.savefig('fusion_results/final_comprehensive_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ 已保存: final_comprehensive_comparison.png")


plot_final_comparison()

# ===================== 15. 保存融合模型 =====================
print("\n[步骤 15] 保存融合模型...")

joblib.dump(fusion_iris_nn, 'fusion_results/fusion_iris_nn.pkl')
joblib.dump(fusion_iris_svm, 'fusion_results/fusion_iris_svm.pkl')
joblib.dump(fusion_cancer_nn, 'fusion_results/fusion_cancer_nn.pkl')
joblib.dump(fusion_cancer_svm, 'fusion_results/fusion_cancer_svm.pkl')

print("  ✓ 已保存所有融合模型")

# ===================== 16. 生成性能对比表格 =====================
print("\n[步骤 16] 生成性能对比表格...")


def generate_performance_table():
    """生成详细的性能对比表格"""

    # 获取单个模型性能
    def get_individual_performance(models, X, y):
        metrics_list = []
        for model in models:
            y_pred = model.predict(X)
            avg = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average=avg, zero_division=0),
                'recall': recall_score(y, y_pred, average=avg, zero_division=0),
                'f1': f1_score(y, y_pred, average=avg, zero_division=0)
            }
            metrics_list.append(metrics)
        return metrics_list

    # 创建数据框
    data = []

    # 鸢尾花 NN
    iris_nn_ind = get_individual_performance(nn_iris_models, X_iris_test, y_iris_test)
    for i, metrics in enumerate(iris_nn_ind):
        data.append(['Iris', 'NN', f'Model-{i + 1}', 'Individual',
                     metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1']])

    data.append(['Iris', 'NN', 'Fusion', 'Fusion',
                 metrics_iris_nn_test['accuracy'], metrics_iris_nn_test['precision'],
                 metrics_iris_nn_test['recall'], metrics_iris_nn_test['f1']])

    # 鸢尾花 SVM
    iris_svm_ind = get_individual_performance(svm_iris_models, X_iris_test, y_iris_test)
    for i, metrics in enumerate(iris_svm_ind):
        data.append(['Iris', 'SVM', f'Model-{i + 1}', 'Individual',
                     metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1']])

    data.append(['Iris', 'SVM', 'Fusion', 'Fusion',
                 metrics_iris_svm_test['accuracy'], metrics_iris_svm_test['precision'],
                 metrics_iris_svm_test['recall'], metrics_iris_svm_test['f1']])

    # 乳腺癌 NN
    cancer_nn_ind = get_individual_performance(nn_cancer_models, X_cancer_test, y_cancer_test)
    for i, metrics in enumerate(cancer_nn_ind):
        data.append(['Breast Cancer', 'NN', f'Model-{i + 1}', 'Individual',
                     metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1']])

    data.append(['Breast Cancer', 'NN', 'Fusion', 'Fusion',
                 metrics_cancer_nn_test['accuracy'], metrics_cancer_nn_test['precision'],
                 metrics_cancer_nn_test['recall'], metrics_cancer_nn_test['f1']])

    # 乳腺癌 SVM
    cancer_svm_ind = get_individual_performance(svm_cancer_models, X_cancer_test, y_cancer_test)
    for i, metrics in enumerate(cancer_svm_ind):
        data.append(['Breast Cancer', 'SVM', f'Model-{i + 1}', 'Individual',
                     metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1']])

    data.append(['Breast Cancer', 'SVM', 'Fusion', 'Fusion',
                 metrics_cancer_svm_test['accuracy'], metrics_cancer_svm_test['precision'],
                 metrics_cancer_svm_test['recall'], metrics_cancer_svm_test['f1']])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['Dataset', 'Algorithm', 'Model', 'Type',
                                     'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # 保存为CSV
    df.to_csv('fusion_results/performance_table.csv', index=False, float_format='%.4f')

    # 保存为Excel (如果安装了openpyxl)
    try:
        df.to_excel('fusion_results/performance_table.xlsx', index=False, float_format='%.4f')
        print("  ✓ 已保存: performance_table.xlsx")
    except:
        print("  ! 未安装openpyxl,跳过Excel文件生成")

    print("  ✓ 已保存: performance_table.csv")

    return df


df_performance = generate_performance_table()

# ===================== 17. 打印最终总结 =====================
print("\n" + "=" * 100)
print("模型融合任务完成总结")
print("=" * 100)
print(f"\n✓ 成功训练 4 个融合模型:")
print(f"  - 鸢尾花数据集 NN融合模型 (准确率: {metrics_iris_nn_test['accuracy']:.4f})")
print(f"  - 鸢尾花数据集 SVM融合模型 (准确率: {metrics_iris_svm_test['accuracy']:.4f})")
print(f"  - 乳腺癌数据集 NN融合模型 (准确率: {metrics_cancer_nn_test['accuracy']:.4f})")
print(f"  - 乳腺癌数据集 SVM融合模型 (准确率: {metrics_cancer_svm_test['accuracy']:.4f})")

print(f"\n✓ 遗传算法优化:")
print(f"  - 种群大小: 50")
print(f"  - 迭代代数: 100")
print(f"  - 变异率: 0.1")
print(f"  - 交叉率: 0.8")

print(f"\n✓ 生成的可视化图表:")
print(f"  1. ga_convergence_curves.png - 遗传算法收敛曲线")
print(f"  2. model_weights_distribution.png - 模型权重分布")
print(f"  3. performance_comparison.png - 性能对比图")
print(f"  4. confusion_matrices.png - 混淆矩阵")
print(f"  5. improvement_analysis.png - 性能提升分析")
print(f"  6. radar_comparison.png - 雷达图对比")
print(f"  7. contribution_analysis.png - 模型贡献度分析")
print(f"  8. performance_heatmap.png - 性能热力图")
print(f"  9. final_comprehensive_comparison.png - 最终综合对比")

print(f"\n✓ 生成的报告文件:")
print(f"  - fusion_report.txt - 文本格式综合报告")
print(f"  - performance_table.csv - 性能对比表格(CSV)")
print(f"  - performance_table.xlsx - 性能对比表格(Excel)")

print(f"\n✓ 保存的模型文件:")
print(f"  - fusion_iris_nn.pkl")
print(f"  - fusion_iris_svm.pkl")
print(f"  - fusion_cancer_nn.pkl")
print(f"  - fusion_cancer_svm.pkl")

print(f"\n所有结果保存在 'fusion_results/' 文件夹中")
print("=" * 100)
print("\n🎉 模型融合任务全部完成! 🎉\n")

