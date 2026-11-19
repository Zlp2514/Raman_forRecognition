"""训练期数据增强：用扰动制造"看不见的批次"。

设计要点（这条决定了增强有没有意义）
    SNV 是"减自身均值、除自身标准差"，因此——

        加常数偏移 c  →  (y+c-mean-c)/std  =  不变
        乘增益 a      →  a(y-mean)/(a·std) =  不变

    也就是说，**在 SNV 之后再靠加偏移、调增益做增强，等于什么都没做**。
    这是很多人做光谱增强时的第一个坑：扩增了数据量，却一个有效样本都没增加。

    真正能穿过 SNV 的扰动只有改变**形状**的那几种：
        baseline_shape  叠加随机低阶多项式/指数形状（模拟荧光背景起伏）
        shift           沿波数整体平移（模拟波数轴校准漂移）
        blur            轻度高斯模糊（模拟峰宽随聚焦/分辨率变化）
        noise           加高斯噪声（模拟光子噪声）

四种扰动可以独立开关，便于做消融，看清是哪一种在起作用。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _row_std(X: np.ndarray) -> np.ndarray:
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return std


def jitter_baseline_shape(X: np.ndarray, rng: np.random.Generator,
                          strength: float = 0.35) -> np.ndarray:
    """叠加随机的低阶多项式形状，模拟每次测量的荧光背景起伏。

    幅度按每条光谱自身的标准差缩放，保证扰动强度与信号量级无关。
    """
    n, p = X.shape
    t = np.linspace(-1.0, 1.0, p)
    std = _row_std(X)
    coeffs = rng.normal(0.0, 1.0, size=(n, 3)) * strength
    shape = (coeffs[:, [0]] * t
             + coeffs[:, [1]] * (t ** 2)
             + coeffs[:, [2]] * (t ** 3))
    return X + shape * std


def jitter_shift(X: np.ndarray, rng: np.random.Generator,
                 max_shift: int = 6) -> np.ndarray:
    """沿波数方向整体平移若干点，模拟波数轴校准漂移。

    用边缘值填充而不是 np.roll，避免把尾部数据卷到头部造成假的强峰。
    """
    n, p = X.shape
    out = np.empty_like(X)
    for i in range(n):
        k = int(rng.integers(-max_shift, max_shift + 1))
        if k == 0:
            out[i] = X[i]
        elif k > 0:
            out[i, :p - k] = X[i, k:]
            out[i, p - k:] = X[i, -1]
        else:
            out[i, -k:] = X[i, :p + k]
            out[i, :-k] = X[i, 0]
    return out


def jitter_blur(X: np.ndarray, rng: np.random.Generator,
                sigma_range: tuple[float, float] = (0.4, 1.2)) -> np.ndarray:
    """轻度高斯模糊，模拟峰宽随聚焦状态与光谱分辨率变化。"""
    n = X.shape[0]
    sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=n)
    return np.vstack([gaussian_filter1d(X[i], sigma=sigmas[i], mode="nearest")
                      for i in range(n)])


def jitter_noise(X: np.ndarray, rng: np.random.Generator,
                 level: float = 0.02) -> np.ndarray:
    """按行标准差的比例叠加高斯噪声。"""
    return X + rng.normal(0.0, 1.0, X.shape) * _row_std(X) * level


def augment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_copies: int = 3,
    seed: int = 0,
    use_baseline: bool = True,
    use_shift: bool = True,
    use_blur: bool = True,
    use_noise: bool = True,
    baseline_strength: float = 0.35,
    max_shift: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把训练集扩增为 n_copies 份扰动副本（含原始数据）。

    返回 (X_aug, y_aug, groups_aug)。原始样本保留，每一份副本独立随机扰动。
    分组标签随样本一起复制，保证后续仍可按批次分析。
    """
    if n_copies < 1:
        raise ValueError("n_copies 至少为 1（1 表示不扩增）")

    rng = np.random.default_rng(seed)
    Xs, ys, gs = [X], [y], [groups]
    for _ in range(n_copies - 1):
        Xc = X.copy()
        if use_baseline:
            Xc = jitter_baseline_shape(Xc, rng, strength=baseline_strength)
        if use_shift:
            Xc = jitter_shift(Xc, rng, max_shift=max_shift)
        if use_blur:
            Xc = jitter_blur(Xc, rng)
        if use_noise:
            Xc = jitter_noise(Xc, rng)
        Xs.append(Xc)
        ys.append(y)
        gs.append(groups)

    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs))
