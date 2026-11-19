"""无真值条件下的基线参数选择。

要解决的问题
    合成数据有真值，可以用基线误差（RMSE）网格搜索挑最优参数；
    真实光谱没有"正确背景"这个标准答案，误差算不出来，于是实践中的做法往往是
    沿用默认参数或凭经验手动调——这正是本项目早期观察到的痛点：
    同一个 IModPoly，多项式阶数取 2 时误差约 13.8，取 5 时 0.727，差近 20 倍。

思路
    既然不能比较"离真值多远"，就转而评估"校正结果本身是否合理"。
    本模块定义四个**不需要真值**的代理指标，各自惩罚一种具体的失败模式：

        neg_ratio           负值比例：扣过头会把峰削成负值
        baseline_curvature  基线曲率：基线应在平滑尺度上变化，跟着峰起伏就是错的
        lowfreq_residual    低频残差：校正谱的缓慢成分应接近 0，同时惩罚欠扣与过扣
        noise_deviation     噪声保持度偏离：减掉一条平滑曲线不应改变高频噪声，
                            偏离 1 说明基线吃掉了噪声（过拟合）或引入抖动

    四个指标**统一为越小越好**，先做秩归一化（抗离群），再等权平均成总分。

    早期版本用过"信噪比"作为第四个指标，这是错的：残留背景会把最大值抬高，
    信噪比反而奖励"欠扣"，实测导致 pspline_asls 选出灾难参数（RMSE 232，最优为 6.8）。
    改为噪声保持度后该问题消失。

能不能信
    代理指标是否可信，不能靠直觉，要用合成数据检验：
    在已知真值的合成谱上，同时算真实 RMSE 与四个代理指标，
    求二者的 Spearman 秩相关。秩相关高（例如 |ρ| > 0.7），
    说明"按代理指标排序"与"按真实误差排序"大体一致，无真值选参才站得住。
    这一步的检验脚本见 12_param_selection.py。
"""
from __future__ import annotations

import numpy as np


def estimate_noise(y: np.ndarray) -> float:
    """用一阶差分的 MAD 估计噪声标准差，抗尖峰与慢变背景干扰。"""
    d = np.diff(np.asarray(y, dtype=float))
    if d.size == 0:
        return 0.0
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def proxy_metrics(y: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    """计算四个无真值代理指标，**全部越小越好**。

    参数
        y         观测光谱（含荧光背景与噪声）
        baseline  估计出的背景曲线
    """
    y = np.asarray(y, dtype=float)
    b = np.asarray(baseline, dtype=float)
    c = y - b

    sigma_c = estimate_noise(c)
    sigma_y = estimate_noise(y)
    eps = 1e-12
    span = float(np.ptp(y)) + eps

    # 1) 负值比例：低于 -2σ 视为真的扣过头（只数噪声以下的负值，不计随机起伏）
    neg_ratio = float(np.mean(c < -2.0 * sigma_c)) if sigma_c > 0 else 0.0

    # 2) 基线曲率：二阶差分的均方根，按原谱幅度归一
    curv = float(np.sqrt(np.mean(np.diff(b, 2) ** 2)) / span) if b.size > 2 else 0.0

    # 3) 低频残差：大窗口移动平均的能量占比
    #    欠扣时残留背景为正的低频成分；过扣时出现负的低频凹陷。两者都会抬高该值。
    w = max(5, c.size // 50)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    low = np.convolve(c, kernel, mode="same")
    lowfreq = float(np.sqrt(np.mean(low ** 2)) / (np.std(c) + eps))

    # 4) 噪声保持度偏离：减掉平滑曲线不应改变高频噪声水平
    noise_dev = float(abs(sigma_c / (sigma_y + eps) - 1.0)) if sigma_y > 0 else 0.0

    return {
        "neg_ratio": neg_ratio,
        "baseline_curvature": curv,
        "lowfreq_residual": lowfreq,
        "noise_deviation": noise_dev,
    }


def _rank_norm(vals: np.ndarray) -> np.ndarray:
    """秩归一化到 [0,1]：抗离群，且对指标量纲不敏感。"""
    n = vals.size
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    if np.ptp(ranks) == 0:
        return np.full(n, 0.5)
    return ranks / (n - 1)


def score_candidates(metrics_list: list[dict[str, float]]) -> np.ndarray:
    """把一组候选参数合成为总分（越大越好）。

    合成规则：**最差项决定得分**，即 `1 - max(各指标秩归一化后的坏度)`。

    为什么不用等权平均：四项指标里，"负值比例""基线曲率""噪声偏离"对**根本不扣背景**
    的解（基线取常数或零）全部给出满分——曲率为零、无负值、噪声不变。
    等权平均会让这类灾难参数总分很高（实测 asls 选中排名 20/20、RMSE 177，最优为 7.0）。
    取 max 后，只要有一项（这里是低频残差）暴露"背景没扣掉"，总分就上不去。

    秩归一化使偏序只反映排序，不受某一指标量级异常放大影响。
    """
    if not metrics_list:
        return np.array([])

    keys = ["neg_ratio", "baseline_curvature", "lowfreq_residual", "noise_deviation"]
    z = np.vstack([_rank_norm(np.array([m[k] for m in metrics_list], dtype=float))
                   for k in keys])
    return 1.0 - np.max(z, axis=0)
