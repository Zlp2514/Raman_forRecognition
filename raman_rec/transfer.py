"""批次效应校正与领域自适应。

要解决什么问题
    同一台仪器、同一批实验，只要换个测量日、换片载玻片，光谱整体形状就会漂移一点。
    P2 的结果显示：模型在这种"没见过的批次"上会掉 9–13 个百分点。
    本模块提供三种**不需要标签**的校正手段——它们在真实部署中都可实现，
    因为推理时你总能先测一批未标注样本、或用该批次自身的统计量做归一化。

三种手段
    derivative     一阶导数。加性基线偏移被微分消掉，乘性缩放被后续 SNV 消掉，
                   代价是噪声放大、峰形变形。
    batch_center   批次中心化。减去所在批次的平均谱，直接消掉"这一批整体长什么样"。
                   只需要该批次的一批未标注光谱，不需要任何标签。
    coral_align    相关对齐（CORAL）。把源域特征的二阶统计量（协方差）白化后
                   重新染色成目标域的，属于经典的浅层域自适应方法。

注意
    这些方法都只使用**特征**，不使用标签，因此不构成数据泄漏。
    但如果测试批次在真实场景里无法一次性拿到（只能单条在线预测），
    batch_center 与 coral 的假设就不成立，这一点在论文里必须交代。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def row_snv(X: np.ndarray) -> np.ndarray:
    """对特征矩阵逐行做标准正态变换（SNV）：减行均值、除以行标准差。

    等价于对每条光谱调用 Spectrum.normalized("snv")，但直接作用在矩阵上更快。
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def derivative(X: np.ndarray, order: int = 1, window: int = 9,
               polyorder: int = 3) -> np.ndarray:
    """沿波数方向求 Savitzky-Golay 导数，同时完成平滑。

    一阶导数把"常数偏移"变成 0，把"缓慢荧光背景"压成接近 0 的缓变项，
    因此对加性批次偏移最直接有效。
    """
    X = np.asarray(X, dtype=float)
    if window % 2 == 0:
        window += 1
    window = min(window, X.shape[1] - (1 - X.shape[1] % 2))
    if window <= polyorder:
        return X.copy()
    return savgol_filter(X, window_length=window, polyorder=polyorder,
                         deriv=order, axis=1)


def batch_center(X: np.ndarray, groups) -> np.ndarray:
    """按批次中心化：每条光谱减去所属批次的平均谱。

    参数
        X       特征矩阵 (n_samples, n_features)
        groups  与 X 行对应的批次标签序列

    只使用同一批次内其他样本的无标签均值，不使用任何标签信息。
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups).astype(str)
    out = X.copy()
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue                     # 单条样本无批次均值可言，保持原样
        out[mask] = X[mask] - X[mask].mean(axis=0, keepdims=True)
    return out


def _sqrtm_psd(C: np.ndarray, inverse: bool = False, eps: float = 1e-6) -> np.ndarray:
    """对称半正定矩阵的平方根（或其逆），用特征分解实现并做数值兜底。"""
    C = (C + C.T) / 2.0
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, eps, None)
    vals = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
    return (vecs * vals) @ vecs.T


def coral_align(Zs: np.ndarray, Zt: np.ndarray,
                eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """CORAL：把源域特征对齐到目标域的协方差与均值。

        Zs' = (Zs - μs) · Cs^(-1/2) · Ct^(1/2) + μt

    参数
        Zs  源域（训练集）特征，通常是 PCA 降维后的低维表示
        Zt  目标域（测试批次）特征，无需标签

    建议在 PCA 之后调用：维度降到几十维再做矩阵平方根，
    既快又稳；在高维原始空间直接做 1376×1376 的特征分解没有必要。
    """
    Zs = np.asarray(Zs, dtype=float)
    Zt = np.asarray(Zt, dtype=float)
    mu_s, mu_t = Zs.mean(axis=0), Zt.mean(axis=0)
    Cs = np.cov(Zs, rowvar=False) + eps * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + eps * np.eye(Zt.shape[1])
    W = _sqrtm_psd(Cs, inverse=True, eps=eps) @ _sqrtm_psd(Ct, eps=eps)
    return (Zs - mu_s) @ W + mu_t, Zt
