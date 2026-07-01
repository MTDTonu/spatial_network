"""
空间预测器 —— 分类 / 回归 / 概率预测 演示
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from spatial_network import SpatialNetwork

print("=" * 60)
print("  空间预测器 —— 分类 · 回归 · 概率")
print("=" * 60)

# 依赖导入
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 导入你的空间预测器模型
from spatial_network import SpatialNetwork

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data    # 4维特征
y = iris.target  # 3类标签：0=setosa,1=versicolor,2=virginica
class_names = iris.target_names

# 2. 划分训练105、测试45（总样本150，固定随机种子复现你截图结果）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
print(f"特征维度: {X_train.shape[1]}")

# 3. 特征标准化（空间邻域距离计算关键）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 初始化你指定参数的SpatialNetwork模型
sn = SpatialNetwork(
    anisotropic=True,              # 各向异性场（高维自动启用）
    regression_method='local_linear',  # 局部线性回归
    kernel='invquad'               # 逆二次核
)

# 5. 模型训练与预测
print("\n===== 开始训练 SpatialNetwork =====")
sn.place(X_train_scaled, y_train)
y_pred = sn.read(X_test_scaled)

# 6. 输出评估结果（和你截图格式对齐）
acc = accuracy_score(y_test, y_pred)
print(f"\n整体准确率: {acc:.2%}")

print("\n===== 分类报告 =====")
print(classification_report(y_test, y_pred, target_names=class_names, digits=2))

print("\n===== 混淆矩阵 =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)



exit(0)
# ═══════════════════════════════════════════════════════
# 一、分类预测
# ═══════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  一、分类预测")
print("─" * 60)

for name, loader in [("Iris（鸢尾花）", load_iris), ("Wine（葡萄酒）", load_wine)]:
    X, y = loader(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    sn = SpatialNetwork()
    sn.place(X_train, y_train)
    y_pred = sn.read(X_test)
    acc = (y_pred == y_test).mean()

    print(f"\n  {name}  |  n={len(X_train)}  d={X.shape[1]}  classes={len(np.unique(y))}")
    print(f"  准确率: {acc:.1%}")
    print(f"  因果荷: [{sn.causal_strength_.min():.1f}, {sn.causal_strength_.max():.1f}]")
    print(f"  场域半径: [{sn.field_radius_.min():.2f}, {sn.field_radius_.max():.2f}]")

# ═══════════════════════════════════════════════════════
# 二、概率预测
# ═══════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  二、概率预测")
print("─" * 60)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

sn = SpatialNetwork()
sn.place(X_train, y_train)

# 取 3 个不同类别的测试样本
proba = sn.predict_proba(X_test)
y_pred = sn.read(X_test)

print(f"\n  前 10 个测试样本的预测概率:")
print(f"  {'样本':>4}  {'真实':>4}  {'预测':>4}", end="")
for c in sn.classes_:
    print(f"  {'P(类'+str(c)+')':>10}", end="")
print()
for i in range(min(10, len(X_test))):
    print(f"  {i:>4}  {y_test[i]:>4}  {y_pred[i]:>4}", end="")
    idx_map = {c: j for j, c in enumerate(sn.classes_)}
    for c in sn.classes_:
        print(f"  {proba[i, idx_map[c]]:>10.4f}", end="")
    print()

# 不确定样本
uncertainty = np.abs(proba.max(axis=1) - 0.5)
boundary_idx = np.argsort(uncertainty)[:3]
print(f"\n  最不确定的 3 个样本（接近边界）:")
for i in boundary_idx:
    print(f"    样本{i}: 真实={y_test[i]}  预测={y_pred[i]}  P_max={proba[i].max():.3f}")

# ═══════════════════════════════════════════════════════
# 三、回归预测
# ═══════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  三、回归预测")
print("─" * 60)

for n, d, noise, tag in [(300, 10, 15, "低维低噪"), (500, 20, 30, "中维中噪")]:
    X, y = make_regression(n, d, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 加权平均（0 阶）
    sn0 = SpatialNetwork(regression_method='weighted_avg')
    sn0.place(X_train, y_train)
    y_pred0 = sn0.read(X_test)

    # 局部线性（1 阶）
    sn1 = SpatialNetwork(regression_method='local_linear')
    sn1.place(X_train, y_train)
    y_pred1 = sn1.read(X_test)

    print(f"\n  {tag} ({d}D, noise={noise})  |  n={len(X_train)}")
    print(f"  加权平均: R²={r2_score(y_test, y_pred0):.4f}  RMSE={np.sqrt(mean_squared_error(y_test, y_pred0)):.2f}")
    print(f"  局部线性: R²={r2_score(y_test, y_pred1):.4f}  RMSE={np.sqrt(mean_squared_error(y_test, y_pred1)):.2f}")

print("\n" + "=" * 60)
print("  完成")
print("=" * 60)
