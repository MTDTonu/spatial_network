"""
空间预测器 (Spatial Predictor)
================================
一个基于场几何的简单机器学习工具。

核心操作：
  place(X, y)        → 放置样本，构建因果场
  read(X)            → 读取场值，预测
  predict_proba(X)   → 输出概率值（softmax）
  place_one(x, y)    → 增量学习

支持：分类 / 回归 / 概率输出 / 各向异性高维 / 边界检测

哲学基础详见：空间预测器——哲学原理.md
完整架构详见：空间预测器——学习指南.md
"""

import numpy as np
from scipy.spatial.distance import cdist


class SpatialNetwork:

    def __init__(self, eps=None, q_scale=5.0, kernel='invquad', anisotropic=True,
                 regression_method='local_linear'):
        self.eps = eps
        self.q_scale = q_scale
        self._eps_computed_ = None
        self.anisotropic = anisotropic
        self.regression_method = regression_method

        if callable(kernel):
            self._kernel_fn = kernel
        else:
            self._kernel_fn = {
                'invquad': lambda d, lam: 1.0 / (1.0 + (d / lam) ** 2),
                'cos': lambda d, lam: np.cos(d / lam),
                'linear': lambda d, lam: np.maximum(0, 1.0 - d / lam),
                'exp': lambda d, lam: np.exp(-d / lam),
            }[kernel]

        self.X_ = None
        self.y_ = None
        self.q_ = None
        self.lambda_ = None
        self.r_ = None
        self.dim_weights_ = None
        self.x_mean_ = None
        self.x_std_ = None

    # ═══════════════════════════════════════════════════════
    # 各向异性：维度权重
    # ═══════════════════════════════════════════════════════

    def _compute_dim_weights(self):
        X, y, n, d = self.X_, self.y_, len(self.X_), self.X_.shape[1]
        if d <= 1:
            self.dim_weights_ = np.ones(d)
            return
        if not self.is_classifier_:
            y_f = y.astype(float)
            corr = np.array([abs(np.corrcoef(X[:, k], y_f)[0, 1]) if not np.isnan(np.corrcoef(X[:, k], y_f)[0, 1]) else 0.0 for k in range(d)])
            corr = np.maximum(corr, 1e-10)
            w = corr / corr.mean()
            self.dim_weights_ = np.clip(w, 0.01, 100.0)
            return
        classes, n_classes = np.unique(y), len(np.unique(y))
        if n_classes < 2:
            self.dim_weights_ = np.ones(d)
            return
        grand_mean = X.mean(axis=0)
        between, within = np.zeros(d), np.zeros(d)
        for cls in classes:
            mask = y == cls
            n_c, cls_mean = mask.sum(), X[mask].mean(axis=0)
            between += n_c * (cls_mean - grand_mean) ** 2
            within += ((X[mask] - cls_mean) ** 2).sum(axis=0)
        eps_f = 1e-10
        between /= max(n_classes - 1, 1)
        within /= max(n - n_classes, 1)
        f_stat = between / (within + eps_f)
        power = 1.0 + np.clip((d - 100) / 400, 0.0, 1.0) * 1.2
        f_powered = f_stat ** power
        w = f_powered / (f_powered.mean() or 1.0)
        self.dim_weights_ = np.clip(w, 0.001, 1000.0)

    def _cdist(self, A, B):
        if self.anisotropic and self.dim_weights_ is not None:
            sqrt_w = np.sqrt(self.dim_weights_)
            return cdist(A * sqrt_w, B * sqrt_w)
        return cdist(A, B)

    # ═══════════════════════════════════════════════════════
    # 核心：place / read
    # ═══════════════════════════════════════════════════════

    def place(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.atleast_1d(np.asarray(y))

        self.is_classifier_ = not np.issubdtype(y.dtype, np.floating)
        if not self.is_classifier_:
            self.is_classifier_ = len(np.unique(y)) < 20 and len(np.unique(y)) > 1
        if self.is_classifier_:
            all_y = np.concatenate([self.y_, y]) if self.X_ is not None else y
            self.classes_ = np.unique(all_y)

        if self.X_ is None:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_ = X.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
        X_norm = (X - self.x_mean_) / self.x_std_

        if self.X_ is None:
            self.X_ = X_norm
            self.y_ = y
        else:
            self.X_ = np.vstack([self.X_, X_norm])
            self.y_ = np.concatenate([self.y_, y])

        self._recompute_all_props()

    def _recompute_all_props(self):
        n = len(self.X_)
        if n == 1:
            self.q_ = np.array([1.0])
            self.lambda_ = np.array([1.0])
            self.r_ = np.array([0.0])
            self.dim_weights_ = np.ones(self.X_.shape[1])
            return

        if self.anisotropic:
            self._compute_dim_weights()
        else:
            self.dim_weights_ = np.ones(self.X_.shape[1])

        d = self.X_.shape[1]
        dists = self._cdist(self.X_, self.X_)
        k = min(10, n - 1)
        knn = np.sort(dists, axis=1)[:, 1:k+1].mean(axis=1)

        if n <= 5:
            self.lambda_ = np.clip(knn, 0.3, np.sqrt(d) * 3)
        else:
            scale = np.median(knn) or 1.0
            blend = np.clip((d - 60) / 90, 0.0, 1.0)
            self.lambda_ = np.clip(knn / scale, 0.3, 5.0) * (1 - blend) + np.clip(knn, 0.3, np.sqrt(d) * 3) * blend

        self.q_ = np.ones(n)
        for i in range(n):
            in_range = dists[i] <= self.lambda_[i]
            in_range[i] = False
            if in_range.any():
                if self.is_classifier_:
                    same = (self.y_[in_range] == self.y_[i]).mean()
                else:
                    yv = self.y_.astype(float)
                    lv = np.var(yv[in_range]) if in_range.sum() >= 2 else 0
                    gv = np.var(yv) or 1.0
                    same = 1.0 / (1.0 + lv / gv)
                self.q_[i] = 1.0 + self.q_scale * same * np.log1p(in_range.sum())

        if self.eps is None:
            d = self.X_.shape[1]
            base_eps = np.median(self.q_) / 10.0
            self._eps_computed_ = base_eps / max(1.0, d / 5.0)
            self._eps_computed_ = max(self._eps_computed_, 0.02)

        self.r_ = self.lambda_ * np.sqrt(np.maximum(self.q_ / self._eps() - 1, 0))

    def _eps(self):
        return self.eps if self.eps is not None else (self._eps_computed_ or 0.5)

    @property
    def epsilon_(self):
        return self._eps()

    @property
    def causal_strength_(self):
        return self.q_

    @property
    def field_radius_(self):
        return self.r_

    def place_one(self, x, y_val):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.X_ is None:
            self.place(x.reshape(1, -1), np.array([y_val]))
            return
        x_norm = (x - self.x_mean_) / self.x_std_
        q, lam, r = self._local_props_from(x_norm, y_val)
        self.X_ = np.vstack([self.X_, x_norm])
        self.y_ = np.append(self.y_, y_val)
        self.q_ = np.append(self.q_, q)
        self.lambda_ = np.append(self.lambda_, lam)
        self.r_ = np.append(self.r_, r)
        dists = self._cdist(x_norm.reshape(1, -1), self.X_[:-1])[0]
        affected = dists <= self.r_[:-1] + r
        for j in np.where(affected)[0]:
            qj, lj, rj = self._local_props_from(self.X_[j], self.y_[j])
            self.q_[j], self.lambda_[j], self.r_[j] = qj, lj, rj

    def _local_props_from(self, x_norm, y_val):
        n = len(self.X_)
        dists = self._cdist(x_norm.reshape(1, -1), self.X_)[0]
        k = min(10, n)
        lam = max(np.mean(np.sort(dists)[:k]), 0.3)
        in_range = dists <= lam
        if in_range.any():
            if self.is_classifier_:
                same = (self.y_[in_range] == y_val).mean()
            else:
                yv = self.y_.astype(float)
                lv = np.var(yv[in_range]) if in_range.sum() >= 2 else 0
                gv = np.var(yv) or 1.0
                same = 1.0 / (1.0 + lv / gv)
            q = 1.0 + self.q_scale * same * np.log1p(in_range.sum())
        else:
            q = 1.0
        r = lam * np.sqrt(max(q / self._eps() - 1, 0))
        return q, lam, r

    # ═══════════════════════════════════════════════════════
    # 读取场值
    # ═══════════════════════════════════════════════════════

    def read(self, X):
        if self.X_ is None:
            raise ValueError("空间中没有样本。先 place()。")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.x_mean_) / self.x_std_

        if self.is_classifier_:
            return self._read_classify(X_norm)
        else:
            return self._read_regression(X_norm)

    def _read_classify(self, X_norm):
        n_classes = len(self.classes_)
        dists = self._cdist(X_norm, self.X_)
        covered = dists <= self.r_[None, :]
        w = np.zeros_like(dists)
        for j in range(len(self.X_)):
            mask = covered[:, j]
            if mask.any():
                w[mask, j] = self.q_[j] * self._kernel_fn(dists[mask, j], self.lambda_[j])
        scores = np.zeros((X_norm.shape[0], n_classes))
        for k, cls in enumerate(self.classes_):
            is_cls = (self.y_ == cls)
            scores[:, k] = w[:, is_cls].sum(axis=1) - w[:, ~is_cls].sum(axis=1) / max(n_classes - 1, 1)
        return self.classes_[np.argmax(scores, axis=1)]

    def _read_regression(self, X_norm):
        dists = self._cdist(X_norm, self.X_)
        if self.regression_method == 'weighted_avg':
            return self._predict_weighted_avg(X_norm, dists)
        else:
            return self._predict_local_linear(X_norm, dists)

    def _predict_weighted_avg(self, X_norm, dists):
        covered = dists <= self.r_[None, :]
        w = np.zeros_like(dists)
        for j in range(len(self.X_)):
            mask = covered[:, j]
            if mask.any():
                w[mask, j] = self.q_[j] * self._kernel_fn(dists[mask, j], self.lambda_[j])
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum < 1e-10] = 1.0
        return (w * self.y_.astype(float)[None, :]).sum(axis=1) / w_sum.sum(axis=1)

    def _predict_local_linear(self, X_norm, dists):
        m, n, d = X_norm.shape[0], len(self.X_), self.X_.shape[1]
        y_f = self.y_.astype(float)
        covered = dists <= self.r_[None, :]
        y_pred = np.zeros(m)
        ridge = 1e-4
        for i in range(m):
            cov_idx = np.where(covered[i])[0]
            k = len(cov_idx)
            if k < d + 2:
                if k == 0:
                    y_pred[i] = np.mean(y_f)
                    continue
                w_local = np.array([self.q_[j] * self._kernel_fn(dists[i, j], self.lambda_[j]) for j in cov_idx])
                y_pred[i] = np.average(y_f[cov_idx], weights=w_local)
                continue
            X_local = self.X_[cov_idx]
            y_local = y_f[cov_idx]
            w = np.array([self.q_[j] * self._kernel_fn(dists[i, j], self.lambda_[j]) for j in cov_idx])
            X_centered = X_local - X_norm[i]
            sqrt_w = np.sqrt(np.maximum(w, 1e-15))
            X_w = X_centered * sqrt_w[:, None]
            y_w = y_local * sqrt_w
            X_design = np.column_stack([sqrt_w, X_w])
            XtX = X_design.T @ X_design
            XtX.flat[::X_design.shape[1] + 1] += ridge
            Xty = X_design.T @ y_w
            try:
                beta = np.linalg.solve(XtX, Xty)
                y_pred[i] = beta[0]
            except np.linalg.LinAlgError:
                y_pred[i] = np.average(y_local, weights=w)
        return y_pred

    # ═══════════════════════════════════════════════════════
    # 概率输出
    # ═══════════════════════════════════════════════════════

    def predict_proba(self, X):
        if self.X_ is None:
            raise ValueError("空间中没有样本。先 place()。")
        if not self.is_classifier_:
            raise ValueError("predict_proba 仅支持分类。")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.x_mean_) / self.x_std_

        n_classes = len(self.classes_)
        dists = self._cdist(X_norm, self.X_)
        covered = dists <= self.r_[None, :]
        w = np.zeros_like(dists)
        for j in range(len(self.X_)):
            mask = covered[:, j]
            if mask.any():
                w[mask, j] = self.q_[j] * self._kernel_fn(dists[mask, j], self.lambda_[j])
        scores = np.zeros((X_norm.shape[0], n_classes))
        for k, cls in enumerate(self.classes_):
            is_cls = (self.y_ == cls)
            scores[:, k] = w[:, is_cls].sum(axis=1) - w[:, ~is_cls].sum(axis=1) / max(n_classes - 1, 1)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # ═══════════════════════════════════════════════════════
    # 边界检测
    # ═══════════════════════════════════════════════════════

    def boundary_score(self, X, n_groups=5):
        if not hasattr(self, 'is_classifier_') or not self.is_classifier_:
            raise ValueError("边界检测仅支持分类")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = (X - self.x_mean_) / self.x_std_
        m, n = X.shape[0], len(self.X_)
        n_grp = min(n_groups, n // 3 + 1)
        idx = np.arange(n)
        groups = [idx[i::n_grp] for i in range(n_grp)]
        all_preds = np.zeros((m, len(groups)), dtype=self.y_.dtype)
        for g, test_idx in enumerate(groups):
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            old_q, old_r = self.q_, self.r_
            self.q_ = old_q.copy() * train_mask
            self.r_ = self.lambda_ * np.sqrt(np.maximum(self.q_ / self._eps() - 1, 0))
            all_preds[:, g] = self.read(X)
            self.q_, self.r_ = old_q, old_r
        n_classes = len(self.classes_)
        scores = np.zeros(m)
        for i in range(m):
            vals, counts = np.unique(all_preds[i], return_counts=True)
            raw = 1.0 - counts.max() / counts.sum()
            max_raw = (n_classes - 1) / n_classes if n_classes > 1 else 1.0
            scores[i] = raw / max_raw if max_raw > 0 else 0.0
        return scores


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    print("=" * 50)
    print("  空间预测器")
    print("=" * 50)

    # 分类
    iris = load_iris()
    Xt, Xe, yt, ye = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)
    sn = SpatialNetwork()
    sn.place(Xt, yt)
    acc = (sn.read(Xe) == ye).mean()
    proba = sn.predict_proba(Xe[:3])
    print(f"\n分类 (Iris): acc={acc:.1%}")
    print(f"前3样本概率:\n{np.array2string(proba, precision=3, suppress_small=True)}")

    # 回归
    Xr, yr = make_regression(200, 10, noise=20, random_state=42)
    Xrt, Xre, yrt, yre = train_test_split(Xr, yr, test_size=0.3, random_state=42)
    sn2 = SpatialNetwork(regression_method='local_linear')
    sn2.place(Xrt, yrt)
    r2 = r2_score(yre, sn2.read(Xre))
    print(f"回归 (合成): R²={r2:.3f}")
