"""开放集识别（Open-set Recognition）：让模型学会说"我不认识"。

为什么需要
    闭集分类假设测试样本一定属于训练时的某一类。现实中这个假设几乎总是不成立：
    测到的可能是库里没有的菌种、掺了别的东西的样品、或者根本不是目标物质。
    此时硬猜的危害大于拒识——药品快检里把假药判成合格，比报"无法判定"危险得多。

本模块提供三种置信度来源，用于把"已知类内排序"转化为"是否为已知"的判断：

    margin       SVM 决策函数最大值。距离所有分类边界都远，说明它不像任何一类。
    prototype    到最近类原型的欧氏距离（标准化空间）。形状最朴素，但可解释。
    mahalanobis  到最近类中心的马氏距离，用类内协方差加权。
                 它考虑了"这个类本身是松散还是紧凑"，通常最可靠。

评估指标
    AUROC        已知与未知在置信度上的可分性，不依赖阈值，最适合比较不同评分器
    闭集准确率    不做拒识时的上限，用来对照
    开放集准确率  未知判为拒识、已知判对且被接受的比例（含拒识代价的真实指标）
    拒识率        在固定接受率（默认已知类保留 95%）下，被判为未知的比例

实现说明
    马氏距离用 Sw^-1 做整体变换后按类中心展开计算，避免对每个类重复做 O(n·d²) 的矩阵乘法。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class OpenSetModel:
    """在已知类上训练，并对任意样本给出"属于已知类"的置信度分数。"""

    def __init__(self, shrink: float = 1e-2):
        self.shrink = shrink

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OpenSetModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)

        self.clf_ = SVC(kernel="linear", C=1.0).fit(Z, y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([Z[y == c].mean(axis=0) for c in self.classes_])

        # 池化类内协方差 + 收缩，理由同 P6：高维下无收缩估计病态
        d = Z.shape[1]
        Sw = np.zeros((d, d))
        dof = 0
        for c in self.classes_:
            Zc = Z[y == c]
            Zc = Zc - Zc.mean(axis=0)
            Sw += Zc.T @ Zc
            dof += Zc.shape[0] - 1
        Sw /= max(dof, 1)
        Sw += self.shrink * (np.trace(Sw) / d) * np.eye(d)
        self.Sw_inv_ = np.linalg.inv(Sw)
        # 预存每个中心的 Sw^-1 c 与其二次型，供展开计算
        self._cent_inv_ = self.centroids_ @ self.Sw_inv_
        self._cent_quad_ = np.einsum("ij,ij->i", self._cent_inv_, self.centroids_)
        return self

    # ---------------------------------------------------------------- 打分
    def score(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """返回三种置信度，数值越大越像"已知类"。"""
        Z = self.scaler_.transform(np.asarray(X, dtype=float))

        margin = self.clf_.decision_function(Z)
        if margin.ndim == 1:                 # 二分类时退化为一维
            margin = np.column_stack([-margin, margin])
        s_margin = margin.max(axis=1)

        d2_proto = ((Z[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        s_proto = -np.sqrt(d2_proto.min(axis=1))

        AZ = Z @ self.Sw_inv_
        quad_x = np.einsum("ij,ij->i", AZ, Z)
        # (x-c)^T Sw^-1 (x-c) = x^T Sw^-1 x - 2 c^T Sw^-1 x + c^T Sw^-1 c
        # 注意 lin 必须用 Z 乘 Sw^-1 c（即 _cent_inv_ 的转置），
        # 早期版本误写成 AZ @ _cent_inv_.T，等于多乘一次 Sw^-1，会把距离算成常数。
        lin = Z @ self._cent_inv_.T
        d2_maha = quad_x[:, None] - 2.0 * lin + self._cent_quad_[None, :]
        s_maha = -np.sqrt(np.clip(d2_maha, 0.0, None).min(axis=1))

        return {"margin": s_margin, "prototype": s_proto, "mahalanobis": s_maha}

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.clf_.predict(Z)


def threshold_at(scores: np.ndarray, keep_rate: float = 0.95) -> float:
    """按"保留多少比例的已知样本"确定阈值。

    取已知样本分数的下分位数：低于它的样本被拒识。
    keep_rate=0.95 表示允许 5% 的已知样本被误拒。
    """
    return float(np.quantile(scores, 1.0 - keep_rate))


def evaluate_openset(
    model: OpenSetModel,
    X_calib: np.ndarray,
    X_known_test: np.ndarray,
    y_known_test: np.ndarray,
    X_unknown_test: np.ndarray,
    keep_rate: float = 0.95,
) -> dict:
    """在一个未知类设定下评估。

    阈值取自**留出的校准集**（训练集中分出来、未参与拟合的那部分），
    不能取自参与拟合的训练样本：拟合样本天然更靠近类中心，分数偏高，
    用它们定分位数会导致测试时过度拒识（实测油脂集拒识率曾达到 100%）。
    """
    s_calib = model.score(X_calib)
    s_known = model.score(X_known_test)
    s_unknown = model.score(X_unknown_test)

    out: dict[str, dict] = {}
    for key in ("margin", "prototype", "mahalanobis"):
        sk, su = s_known[key], s_unknown[key]
        auc = float(roc_auc_score(
            np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su]))

        tau = threshold_at(s_calib[key], keep_rate)
        accepted = sk >= tau
        correct = model.predict(X_known_test) == y_known_test
        known_acc = float(np.mean(correct & accepted))
        reject_rate = float(np.mean(su < tau))

        out[key] = {
            "auroc": auc,
            "closed_acc": float(np.mean(correct)),
            "open_acc": known_acc,
            "reject_rate": reject_rate,
            "threshold": tau,
            "known_accept": float(np.mean(accepted)),
        }

    # 整体开放集准确率：已知判对且接受 + 未知被拒识，按样本数加权
    n_total = len(y_known_test) + len(X_unknown_test)
    for key in out:
        known_part = out[key]["open_acc"] * len(y_known_test)
        unknown_part = out[key]["reject_rate"] * len(X_unknown_test)
        out[key]["open_acc_all"] = float((known_part + unknown_part) / n_total)

    return out
