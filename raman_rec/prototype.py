"""原型分类：用"每类中心"代替判别边界。

为什么值得单独试一次
    SVM 学的是**判别边界**——它在特征空间里划超平面，边界的形状由支持向量决定。
    当训练数据来自多个批次时，支持向量往往来自"最不像其他批次的那些样本"，
    于是边界被批次偏移拉扯，换个批次就失效。

    原型分类学的是**类中心**——每个类别在特征空间里有一个代表点，判据是"离谁近"。
    它没有边界概念，因此对批次造成的整体平移不敏感（只要平移不改变相对位置）。

    这是一个非常便宜的对照：如果原型法明显好于 SVM，
    说明瓶颈在决策边界而非特征表示，后续做度量学习就有的放矢。

四种变体
    centroid       PCA 空间里的欧氏最近中心
    centroid_cos   余弦最近中心（只看方向，忽略幅值）
    centroid_white PCA 分量先白化再加权（消除 PC1 方差过大导致的偏置）
    knn            最近邻（k=5），局部原型
"""
from __future__ import annotations

import numpy as np


class NearestPrototype:
    """最近类原型分类器。

    参数
        metric   "euclidean" 或 "cosine"
        whiten   True 时把每个特征维度除以其标准差，消除主成分方差递减造成的偏置
    """

    def __init__(self, metric: str = "euclidean", whiten: bool = False,
                 eps: float = 1e-8):
        self.metric = metric
        self.whiten = whiten
        self.eps = eps

    def _prep(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            Z = Z / (self._scale + self.eps)
        if self.metric == "cosine":
            norm = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = Z / np.maximum(norm, self.eps)
        return Z

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "NearestPrototype":
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y)
        self._scale = Z.std(axis=0) if self.whiten else np.ones(Z.shape[1])

        Zp = self._prep(Z)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([Zp[y == c].mean(axis=0) for c in self.classes_])
        if self.metric == "cosine":
            norm = np.linalg.norm(self.centroids_, axis=1, keepdims=True)
            self.centroids_ = self.centroids_ / np.maximum(norm, self.eps)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        Zp = self._prep(np.asarray(Z, dtype=float))
        if self.metric == "cosine":
            sim = Zp @ self.centroids_.T          # 余弦相似度，越大越近
            idx = np.argmax(sim, axis=1)
        else:
            d2 = ((Zp[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d2, axis=1)
        return self.classes_[idx]
