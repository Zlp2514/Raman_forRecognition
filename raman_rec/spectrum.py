"""统一的光谱数据结构与通用预处理操作。

一条拉曼光谱由两部分组成：横坐标 x（拉曼位移，cm⁻¹）和纵坐标 y（强度）。
本模块把这两条数组封装成 Spectrum，并把常用操作做成方法，使处理流程可以写成：

    spec = load_csv("my_raman.csv")
    spec = remove_baseline(spec, method="imodpoly", poly_order=5)
    spec = spec.smoothed(window=7)
    spec = spec.normalized("snv")
    peaks = spec.peaks()

每一步都返回新的 Spectrum，不修改原对象。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Spectrum:
    """一条拉曼光谱。

    属性
        x      拉曼位移（cm⁻¹），一维浮点数组
        y      强度，一维浮点数组，与 x 等长
        label  类别标签（用于识别任务），可为 None
        meta   附加信息字典，例如合成数据的真值、采集参数、文件名

    构造时会自动按 x 升序排序并检查数据合法性，因此后续操作可以假设 x 单调递增。
    """

    x: np.ndarray
    y: np.ndarray
    label: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        if self.x.size != self.y.size:
            raise ValueError(f"x 与 y 长度不一致：{self.x.size} vs {self.y.size}")
        if self.x.size < 2:
            raise ValueError("光谱至少需要 2 个点")

        order = np.argsort(self.x)
        if not np.all(order == np.arange(self.x.size)):
            self.x = self.x[order]
            self.y = self.y[order]

        if np.any(np.diff(self.x) <= 0):
            raise ValueError("拉曼位移存在重复值或非单调，请检查数据")
        if not np.all(np.isfinite(self.y)):
            raise ValueError("强度数组含 NaN 或无穷值，请先清理")

    # ---------------------------------------------------------------- 基础属性
    def __len__(self) -> int:
        return int(self.x.size)

    @property
    def span(self) -> float:
        """横坐标跨度（cm⁻¹）。"""
        return float(self.x[-1] - self.x[0])

    @property
    def step(self) -> float:
        """相邻点的平均间距（cm⁻¹），用于把点数换算成峰宽。"""
        return float(np.mean(np.diff(self.x)))

    def copy(self) -> "Spectrum":
        return replace(self, x=self.x.copy(), y=self.y.copy(), meta=dict(self.meta))

    # ---------------------------------------------------------------- 截取与对齐
    def slice(self, low: float | None = None, high: float | None = None) -> "Spectrum":
        """按拉曼位移区间截取，用于裁掉两端噪声大的区域。"""
        mask = np.ones_like(self.x, dtype=bool)
        if low is not None:
            mask &= self.x >= low
        if high is not None:
            mask &= self.x <= high
        if not mask.any():
            raise ValueError(f"区间 [{low}, {high}] 内没有数据点")
        return replace(self, x=self.x[mask], y=self.y[mask], meta=dict(self.meta))

    # ---------------------------------------------------------------- 基本处理
    def smoothed(self, window: int = 7, polyorder: int = 3) -> "Spectrum":
        """Savitzky-Golay 平滑，在不明显压低峰高的前提下抑制高频噪声。

        window 必须是大于 polyorder 的奇数；点数不足时原样返回。
        """
        from scipy.signal import savgol_filter

        window = int(window)
        if window % 2 == 0:
            window += 1
        if window < 3 or window > self.x.size:
            return self.copy()
        if polyorder >= window:
            polyorder = max(1, window - 2)
        return replace(self, y=savgol_filter(self.y, window, polyorder),
                       meta=dict(self.meta))

    def normalized(self, mode: str = "snv") -> "Spectrum":
        """强度归一化，消除激发功率、积分时间、样品量带来的整体幅值差异。

        可选模式
            max     除以最大绝对值，保留相对比例
            minmax  线性拉伸到 [0, 1]
            area    除以积分面积（梯形法），适合比较峰面积占比
            l2      除以 L2 范数，向量夹角意义下可比
            snv     Standard Normal Variate：减均值除标准差，同时消除基线漂移与幅值差异
        """
        y = self.y.astype(float)
        if mode == "max":
            scale = float(np.max(np.abs(y))) or 1.0
            out = y / scale
        elif mode == "minmax":
            lo, hi = float(np.min(y)), float(np.max(y))
            span = hi - lo
            out = (y - lo) / span if span else np.zeros_like(y)
        elif mode == "area":
            area = float(np.trapezoid(np.abs(y), self.x)) if hasattr(np, "trapezoid") \
                else float(np.trapz(np.abs(y), self.x))
            out = y / area if area else y
        elif mode == "l2":
            norm = float(np.linalg.norm(y)) or 1.0
            out = y / norm
        elif mode == "snv":
            std = float(np.std(y))
            out = (y - float(np.mean(y))) / std if std else y - float(np.mean(y))
        else:
            raise ValueError(
                f"未知的归一化模式 {mode!r}，可选：max / minmax / area / l2 / snv")
        return replace(self, y=out, meta=dict(self.meta))

    # ---------------------------------------------------------------- 寻峰
    def peaks(
        self,
        prominence_frac: float = 0.05,
        distance_cm: float | None = None,
        height_frac: float | None = None,
    ) -> dict:
        """检测拉曼峰。

        参数
            prominence_frac  显著度阈值，取 (max-min) 的百分比；0.05 表示至少占 5%
            distance_cm      两个峰之间的最小间距（cm⁻¹），用于避免同一峰检出多个尖
            height_frac      绝对高度阈值，取 (max-min) 的百分比，用于滤掉小鼓包

        返回
            {"index": 索引数组, "shift": 峰位 cm⁻¹, "height": 峰高, "prominence": 显著度}
        """
        from scipy.signal import find_peaks

        span = float(np.max(self.y) - np.min(self.y))
        if span <= 0:
            empty = np.array([], dtype=int)
            return {"index": empty, "shift": np.array([]), "height": np.array([]),
                    "prominence": np.array([])}

        kwargs = {"prominence": prominence_frac * span}
        if distance_cm is not None and distance_cm > 0:
            kwargs["distance"] = max(1, int(round(distance_cm / self.step)))
        if height_frac is not None:
            kwargs["height"] = float(np.min(self.y)) + height_frac * span

        idx, props = find_peaks(self.y, **kwargs)
        return {
            "index": idx,
            "shift": self.x[idx],
            "height": self.y[idx],
            "prominence": props.get("prominences", np.array([])),
        }

    # ---------------------------------------------------------------- 质量评估
    def snr(self) -> float:
        """粗估信噪比。

        噪声用相邻点一阶差分的稳健标准差估计（对窄峰不敏感），
        信号用峰峰值。数值越大表示谱越干净，可用于筛选低质量光谱。
        """
        if self.x.size < 3:
            return float("nan")
        noise = float(np.std(np.diff(self.y)) / np.sqrt(2))
        if noise == 0:
            return float("inf")
        signal = float(np.max(self.y) - np.min(self.y))
        return signal / noise

    def negative_ratio(self, tol: float = 0.0) -> float:
        """低于 -tol 的点所占比例。

        基线校正做得过头的典型症状是出现大片负值，这个指标常用于无真值时的
        参数筛选：负值比例过高说明基线被高估。
        """
        return float(np.mean(self.y < -abs(tol)))

    # ---------------------------------------------------------------- 展示
    def __repr__(self) -> str:
        label = f", label={self.label!r}" if self.label is not None else ""
        return (f"Spectrum(n={len(self)}, "
                f"{self.x[0]:.1f}~{self.x[-1]:.1f} cm-1{label})")
