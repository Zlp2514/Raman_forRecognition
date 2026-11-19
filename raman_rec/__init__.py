"""拉曼光谱识别工具包。

模块划分
    spectrum   —— 统一的光谱数据结构 Spectrum（含归一化、平滑、寻峰、信噪比）
    preprocess —— 荧光背景扣除（基线校正）的统一入口，支持六种方法
    io         —— 两列 CSV 的读写（自动处理编码与表头）
    synthetic  —— 带真值的合成光谱生成器，用于方法验证

设计原则
    1. 所有处理都以 Spectrum 为中心，输入输出都是 Spectrum，便于链式调用。
    2. 基线校正方法只在 preprocess 里定义一份，脚本不再各写一套。
    3. 真值只存在于合成光谱的 meta 里，处理链本身不假设有真值。
"""
from .spectrum import Spectrum
from .io import load_csv, save_csv
from .synthetic import make_synthetic
from .preprocess import (
    BASELINE_METHODS,
    DEFAULT_PARAMS,
    PARAM_GRIDS,
    estimate_baseline,
    remove_baseline,
)
from .datasets import (
    Dataset,
    load_bacterial,
    load_oil,
    stratified_split,
    official_split,
    group_split,
)

__version__ = "0.2.0"

__all__ = [
    "Spectrum",
    "load_csv",
    "save_csv",
    "make_synthetic",
    "BASELINE_METHODS",
    "DEFAULT_PARAMS",
    "PARAM_GRIDS",
    "estimate_baseline",
    "remove_baseline",
    "Dataset",
    "load_bacterial",
    "load_oil",
    "stratified_split",
    "official_split",
    "group_split",
]
