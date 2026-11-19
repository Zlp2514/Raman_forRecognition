"""在同一个拉曼光谱上比较 pybaselines 的六种基线（荧光扣除）方法。

为什么用合成光谱
    只有合成数据才有"真实背景"这个已知答案，才能算出基线误差 RMSE。
    真实实验数据没有真值，只能用峰形是否被削、负值比例等间接指标判断。

用法
    1) IDLE：双击 start_idle.bat，打开本文件按 F5
    2) PowerShell 默认（合成谱）：
       .\\.venv\\Scripts\\python.exe .\\raman_reproduction\\compare_baseline_methods.py
    3) 处理自己的 CSV（无真值，只出基线与统计）：
       .\\.venv\\Scripts\\python.exe .\\raman_reproduction\\compare_baseline_methods.py --csv .\\raman_reproduction\\my_raman.csv

输出（与 01、02 一致，直接写在本脚本所在目录）
    compare_result.png     基线叠加 + 校正谱叠加 + 基线误差对比
    compare_result.csv     每个方法的最优参数、误差与耗时
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pybaselines

from raman_rec import (
    DEFAULT_PARAMS,
    PARAM_GRIDS,
    estimate_baseline,
    make_synthetic,
)
from raman_rec import load_csv as load_spectrum_csv

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT

FIGURE_NAME = "compare_result.png"
TABLE_NAME = "compare_result.csv"

# 参与对比的六种方法（顺序即输出顺序；ModPoly 只在 01_quickstart.py 里作对照）。
# 参数网格与默认参数统一由 raman_rec.preprocess 提供，避免各脚本各写一套。
COMPARE_METHODS = ("imodpoly", "asls", "airpls", "snip",
                   "mixture_model", "pspline_asls")
METHODS = {name: PARAM_GRIDS[name] for name in COMPARE_METHODS}

COLORS = {
    "imodpoly": "#d62728",
    "asls": "#1f77b4",
    "airpls": "#2ca02c",
    "snip": "#ff7f0e",
    "mixture_model": "#9467bd",
    "pspline_asls": "#17becf",
}

# 真实数据没有真值，无法按误差自动挑参，因此改用一组固定推荐值。
# 这些值只是合理起点，不是最优解；需要结合峰形人工微调。
DEFAULTS = {name: DEFAULT_PARAMS[name] for name in COMPARE_METHODS}


def setup_fonts() -> None:
    """让图中的中文正常显示，负号不乱码。"""
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_synthetic():
    """与 01_quickstart.py 完全一致：1600 点，200-2000 cm-1，同一随机种子。"""
    spec = make_synthetic()
    return spec.x, spec.y, spec.meta["true_baseline"], spec.meta["true_signal"]


def load_csv(path: Path):
    """读两列 CSV：第一列拉曼位移，第二列强度（编码与表头由 raman_rec 处理）。"""
    spec = load_spectrum_csv(path)
    x, y = spec.x, spec.y
    if x.size < 20:
        raise ValueError(f"只有 {x.size} 个点，至少需要 20 个才能做对比")
    return x, y


def fit_once(x: np.ndarray, y: np.ndarray, name: str, kwargs: dict):
    start = time.perf_counter()
    baseline, info = estimate_baseline(x, y, name, **kwargs)
    elapsed = time.perf_counter() - start
    return baseline, info["iterations"], elapsed


def describe(kwargs: dict) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in kwargs.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="比较 pybaselines 的六种基线方法")
    parser.add_argument("--csv", type=Path, default=None,
                        help="两列 CSV；不填则使用内置合成光谱")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT,
                        help=f"输出目录，默认 {DEFAULT_OUT}")
    parser.add_argument("--no-show", action="store_true",
                        help="只保存图片，不弹出窗口（批量运行时用）")
    args = parser.parse_args()

    setup_fonts()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.csv is None:
        x, y, true_baseline, true_signal = load_synthetic()
        has_truth = True
        source = "内置合成光谱（1600 点，已知真实背景）"
    else:
        x, y = load_csv(args.csv)
        true_baseline = true_signal = None
        has_truth = False
        source = f"{args.csv}（{x.size} 点，无真值）"

    print("=" * 72)
    print("pybaselines:", pybaselines.__version__)
    print("Python    :", sys.executable)
    print("数据来源  :", source)
    if not has_truth:
        print("选参方式  : 固定推荐参数（无真值，不做自动选参）")
    print("=" * 72)
    param_header = "最优参数" if has_truth else "使用参数"
    print(f"{'方法':<16}{param_header:<30}{'基线RMSE':>10}{'迭代':>6}{'秒':>7}")

    results = []
    for name, grid in METHODS.items():
        if not has_truth:
            grid = [DEFAULTS[name]]
        best = None
        for kwargs in grid:
            try:
                baseline, iters, elapsed = fit_once(x, y, name, kwargs)
            except Exception as exc:               # 参数越界等，跳过该组
                print(f"  [跳过] {name} {kwargs}: {type(exc).__name__}: {exc}")
                continue
            if has_truth:
                score = float(np.sqrt(np.mean((baseline - true_baseline) ** 2)))
            else:
                score = float("nan")
            item = {"method": name, "kwargs": kwargs, "baseline": baseline,
                    "rmse": score, "iters": iters, "seconds": elapsed}
            if best is None or (has_truth and score < best["rmse"]):
                best = item
        if best is None:
            print(f"  [失败] {name}: 所有参数组合均未成功")
            continue
        results.append(best)
        shown = f"{best['rmse']:.3f}" if has_truth else "-"
        print(f"{name:<16}{describe(best['kwargs']):<30}{shown:>10}"
              f"{best['iters']:>6}{best['seconds']:>7.2f}")

    if not results:
        print("没有任何方法成功运行。")
        return 1

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 结果表
    header = "method,params,baseline_rmse,iterations,seconds"
    rows = []
    for item in results:
        rmse = f"{item['rmse']:.6f}" if has_truth else ""
        rows.append(f"{item['method']},\"{describe(item['kwargs'])}\",{rmse},"
                    f"{item['iters']},{item['seconds']:.3f}")
    (outdir / TABLE_NAME).write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    # 绘图：上=基线叠加，中=校正谱叠加，下=误差对比
    if has_truth:
        fig, axes = plt.subplots(
            3, 1, figsize=(11, 11.5), sharex=False,
            gridspec_kw={"height_ratios": [1.15, 1.15, 0.9]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    ax = axes[0]
    ax.plot(x, y, color="0.6", lw=0.9, label="实测谱（含荧光）")
    if has_truth:
        ax.plot(x, true_baseline, "k--", lw=1.8, label="真实背景")
    for item in results:
        ax.plot(x, item["baseline"], color=COLORS[item["method"]], lw=1.3,
                label=f"{item['method']}")
    ax.set_ylabel("强度")
    ax.set_title("六种方法估计的基线", fontsize=12)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() * 0.96, y.max() * 1.30)
    ax.legend(ncol=2, fontsize=8.5, loc="upper center", framealpha=0.94)

    ax = axes[1]
    if has_truth:
        ax.plot(x, true_signal, "k--", lw=1.8, label="真实拉曼信号")
    for item in results:
        ax.plot(x, y - item["baseline"], color=COLORS[item["method"]],
                lw=1.1, label=item["method"])
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("扣除后强度")
    ax.set_title("扣除荧光后的光谱", fontsize=12)
    corrected_all = [y - item["baseline"] for item in results]
    top = max(float(np.max(c)) for c in corrected_all)
    bottom = min(0.0, min(float(np.min(c)) for c in corrected_all))
    ax.set_ylim(bottom - 12, top * 1.38)
    ax.legend(ncol=2, fontsize=8.5, loc="upper center", framealpha=0.94)
    ax.set_xlim(x.min(), x.max())

    if has_truth:
        ax = axes[2]
        names = [i["method"] for i in results]
        values = [i["rmse"] for i in results]
        order = np.argsort(values)[::-1]
        ax.barh([names[i] for i in order], [values[i] for i in order],
                color=[COLORS[names[i]] for i in order], height=0.6)
        for pos, idx in enumerate(order):
            ax.text(values[idx], pos, f"  {values[idx]:.3f}",
                    va="center", fontsize=9)
        ax.set_xlabel("基线 RMSE（相对已知真实背景，越小越好）")
        ax.set_title("误差对比（仅合成数据可用）", fontsize=12)
        ax.set_xlim(0, max(values) * 1.2)

    fig.tight_layout()
    figure_path = outdir / FIGURE_NAME
    fig.savefig(figure_path, dpi=170)

    print("-" * 72)
    print("图  :", figure_path)
    print("表  :", outdir / TABLE_NAME)
    if has_truth:
        best = min(results, key=lambda i: i["rmse"])
        worst = max(results, key=lambda i: i["rmse"])
        print(f"最优: {best['method']} ({describe(best['kwargs'])}) "
              f"RMSE={best['rmse']:.3f}")
        print(f"最差: {worst['method']} ({describe(worst['kwargs'])}) "
              f"RMSE={worst['rmse']:.3f}")
        print("注意: 这里按真值挑参数，真实数据没有真值，不能这样自动选。")
    else:
        print("无真值：图只用于比较基线的形状与位置，不做精度排名。")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
