"""荧光背景扣除（基线校正）的统一入口。

为什么单独一个模块
    基线校正是后续所有环节的前提，但它有六种以上互不通用参数体系的算法。
    把这些算法的调用方式、默认参数、候选参数网格集中在这里，脚本里就不会
    各写一套，换方法或换参数时只改一处。

两类参数的含义完全不同，不要混用
    DEFAULT_PARAMS  真实数据用。真实光谱没有真值，不能按误差挑最优，
                    只能用一组经验上较稳的固定参数。
    PARAM_GRIDS     合成数据用。有真值才能算基线误差、才能网格搜索挑最优。
                    这套"自动挑最优"是合成数据的特权，不能照搬到真实数据。
"""
from __future__ import annotations

import numpy as np
from pybaselines import Baseline

from .spectrum import Spectrum

# 支持的基线校正方法（均为 pybaselines 的经典算法）
BASELINE_METHODS: tuple[str, ...] = (
    "modpoly",         # 迭代多项式拟合（IModPoly 的前身，作为对照基线）
    "imodpoly",        # 迭代多项式拟合，背景接近低阶多项式时最准
    "asls",            # 非对称最小二乘，无多项式假设
    "airpls",          # 自适应迭代重加权，无需人工设定权重
    "snip",            # 统计迭代削峰，不依赖拟合模型
    "mixture_model",   # 混合模型，对复杂背景适应性强
    "pspline_asls",    # 惩罚样条版 asls，适合非平滑背景
)

# 真实数据的推荐起点（不是最优解，需要结合峰形人工微调）
DEFAULT_PARAMS: dict[str, dict] = {
    "modpoly": {"poly_order": 5},
    "imodpoly": {"poly_order": 5},
    "asls": {"lam": 1e6, "p": 0.01},
    "airpls": {"lam": 1e6},
    "snip": {"max_half_window": 40},
    "mixture_model": {"p": 0.5},
    "pspline_asls": {"lam": 1e2, "p": 0.01},
}

# 合成数据上的参数搜索网格
PARAM_GRIDS: dict[str, list[dict]] = {
    "modpoly": [{"poly_order": o} for o in (2, 3, 4, 5, 6, 7, 8)],
    "imodpoly": [{"poly_order": o} for o in (2, 3, 4, 5, 6, 7, 8)],
    "asls": [{"lam": lam, "p": p}
             for lam in (1e4, 1e5, 1e6, 1e7, 1e8)
             for p in (0.001, 0.01, 0.05, 0.1)],
    "airpls": [{"lam": lam} for lam in (1e4, 1e5, 1e6, 1e7, 1e8)],
    "snip": [{"max_half_window": w} for w in (5, 10, 20, 40, 60, 80, 120)],
    "mixture_model": [{"p": p} for p in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9)],
    "pspline_asls": [{"lam": lam, "p": p}
                     for lam in (1e2, 1e3, 1e4, 1e5, 1e6)
                     for p in (0.001, 0.01, 0.05)],
}


def _check_method(method: str) -> None:
    if method not in BASELINE_METHODS:
        raise ValueError(
            f"未知的基线方法 {method!r}，可选：{', '.join(BASELINE_METHODS)}")


def estimate_baseline(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "imodpoly",
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """估计荧光背景曲线。

    参数
        x, y     拉曼位移与强度（y 应已扣除暗电流、宇宙射线等粗差）
        method   BASELINE_METHODS 中的一种
        kwargs   该方法的参数，缺省时使用 DEFAULT_PARAMS 的推荐值

    返回
        (baseline, info)
        baseline  与 y 等长的背景估计
        info      含 params（实际使用的参数）与 iterations（迭代次数）
    """
    _check_method(method)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    params = dict(DEFAULT_PARAMS[method])
    params.update(kwargs)

    fitter = Baseline(x_data=x)
    baseline, fit_info = getattr(fitter, method)(y, **params)
    history = fit_info.get("tol_history", []) if isinstance(fit_info, dict) else []
    return np.asarray(baseline, dtype=float), {
        "params": params,
        "iterations": len(history),
    }


def remove_baseline(spectrum: Spectrum, method: str = "imodpoly", **kwargs) -> Spectrum:
    """扣除荧光背景，返回校正后的新光谱。

    估计出的基线与真值（若有）会存进 meta，便于画图和评估：
        meta["baseline"]       本次估计的背景曲线
        meta["baseline_method"] 方法名
        meta["baseline_params"] 实际参数
    """
    baseline, info = estimate_baseline(spectrum.x, spectrum.y, method, **kwargs)
    meta = dict(spectrum.meta)
    meta["baseline"] = baseline
    meta["baseline_method"] = method
    meta["baseline_params"] = info["params"]
    meta["baseline_iterations"] = info["iterations"]
    return Spectrum(x=spectrum.x, y=spectrum.y - baseline,
                    label=spectrum.label, meta=meta)


def rmse_against_truth(baseline: np.ndarray, truth: np.ndarray) -> float:
    """基线误差（仅合成数据可用）。真实数据没有 truth，不要调用。"""
    baseline = np.asarray(baseline, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((baseline - truth) ** 2)))
