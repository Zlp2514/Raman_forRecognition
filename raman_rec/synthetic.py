"""带真值的合成拉曼光谱生成器。

为什么需要它
    真实光谱没有"真实背景"这个标准答案，就无法定量评价基线校正准不准。
    合成光谱把信号、背景、噪声三部分分开构造，三者都留存在 meta 里，
    于是可以算基线误差 RMSE，也可以算识别任务里的分类准确率。

模型
    y = 拉曼峰（若干高斯） + 荧光背景（指数衰减 + 二次项） + 高斯噪声

    指数项模拟荧光随拉曼位移快速衰减的部分，二次项模拟缓慢弯曲的部分，
    两者叠加得到的背景形状与实测光谱较为接近，且不是低阶多项式——
    这一点很重要：若背景本身就是多项式，多项式类方法会占不合理的便宜。
"""
from __future__ import annotations

import numpy as np
from pybaselines import utils

from .spectrum import Spectrum

# 默认参数与 01_quickstart.py 保持一致，保证历史结果可复现
DEFAULT_SEED = 20261114
DEFAULT_POINTS = 1600
DEFAULT_RANGE = (200.0, 2000.0)
DEFAULT_NOISE = 5.0

# 四个拉曼峰：(幅度, 中心 cm⁻¹, 宽度 cm⁻¹)
DEFAULT_PEAKS = (
    (180.0, 520.0, 10.0),
    (260.0, 1000.0, 16.0),
    (150.0, 1450.0, 12.0),
    (210.0, 1600.0, 20.0),
)


def make_synthetic(
    seed: int = DEFAULT_SEED,
    n_points: int = DEFAULT_POINTS,
    x_range: tuple[float, float] = DEFAULT_RANGE,
    noise_std: float = DEFAULT_NOISE,
    peaks: tuple[tuple[float, float, float], ...] = DEFAULT_PEAKS,
    label: str | None = None,
) -> Spectrum:
    """生成一条合成光谱。

    返回的 Spectrum.meta 中保留三项与真值有关的信息：
        true_signal    不含背景、不含噪声的纯拉曼信号
        true_baseline  真实的荧光背景
        noise_std      噪声标准差
    后续做基线校正时，把估计出的基线与 meta["true_baseline"] 比较即可得到误差。

    修改 seed 会得到不同的噪声实现，但信号与背景不变；
    修改 peaks 可以模拟不同分子，为以后的分类实验准备多类样本。
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x_range[0], x_range[1], int(n_points))

    signal = np.zeros_like(x)
    for amplitude, center, width in peaks:
        signal += utils.gaussian(x, amplitude, center, width)

    # 背景参数刻意写成固定常数而非按区间比例换算：数值一变，
    # 历史实验里那些 RMSE（0.727 等）就再也复现不出来。
    baseline = (250.0
                + 900.0 * np.exp(-(x - 200.0) / 650.0)
                + 0.00035 * (x - 1000.0) ** 2)

    noise = rng.normal(0.0, noise_std, x.size)
    y = signal + baseline + noise

    return Spectrum(
        x=x,
        y=y,
        label=label,
        meta={
            "true_signal": signal,
            "true_baseline": baseline,
            "noise_std": float(noise_std),
            "seed": int(seed),
            "synthetic": True,
        },
    )
