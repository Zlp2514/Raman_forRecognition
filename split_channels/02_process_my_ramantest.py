"""处理 my_ramantest.csv —— 用自己填的数据产出「荧光 + 拉曼」两条曲线和结果图。

用法
    1) 双击运行 / 在 IDLE 或 VS Code 中直接运行本文件（不需要参数）
    2) 命令行：
       python 02_process_my_ramantest.py
       python 02_process_my_ramantest.py --order 6 --method asls
       python 02_process_my_ramantest.py --input 别的文件.csv
       python 02_process_my_ramantest.py --no-show

输入
    与本脚本同目录的 my_ramantest.csv：两列，第一行表头
        raman_shift_cm-1,intensity
        200,1250
        205,1256
    · 第一列拉曼位移，必须从小到大、不能重复
    · 第二列强度，填原始 counts，不要先做基线扣除或归一化
    · 至少 12 个点；建议 300 点以上（点太少时脚本会自动降阶并提示）

输出（写在本脚本所在目录）
    my_ramantest_channels.csv   四列：raman_shift, intensity, fluorescence, raman
    my_ramantest_result.png     三联图：原谱+荧光 / 荧光通道 / 拉曼通道
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 同目录的核心模块可能被改名（例如加上 01_ 前缀），两种都尝试，
# 这样单独改文件名不会把本脚本弄挂。
import importlib

_shared = None
for _cand in ("01_split_two_channels", "split_two_channels"):
    try:
        _shared = importlib.import_module(_cand)
        break
    except ImportError:
        continue

if _shared is None:
    print("找不到同目录的核心模块。")
    print("需要以下任意一个文件与本脚本放在一起：")
    print("  split_two_channels.py  或  01_split_two_channels.py")
    print("当前目录：", HERE)
    raise SystemExit(1)

load_csv = _shared.load_csv
split_channels = _shared.split_channels
qc_metrics = _shared.qc_metrics
verdict = _shared.verdict
setup_fonts = _shared.setup_fonts
plot_channels = _shared.plot_channels

DEFAULT_CSV = HERE / "my_ramantest.csv"
OUT_CSV = HERE / "my_ramantest_channels.csv"
OUT_PNG = HERE / "my_ramantest_result.png"


def auto_order(n, requested):
    """点太少时高阶多项式会穿过每个点，按点数自动降阶。"""
    if requested:
        return requested, None
    if n >= 80:
        return 5, None
    order = max(2, n // 8)
    return order, f"只有 {n} 个点，阶数自动降为 {order}（点少时高阶会过拟合）"


def main() -> int:
    ap = argparse.ArgumentParser(description="处理 my_ramantest.csv")
    ap.add_argument("--input", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--method", default="imodpoly",
                    choices=("imodpoly", "modpoly", "asls", "airpls", "snip"))
    ap.add_argument("--order", type=int, default=None,
                    help="多项式阶数；不填则按点数自动选择")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    setup_fonts()

    src = args.input if args.input.is_absolute() else (Path.cwd() / args.input)
    if not src.exists():
        print(f"找不到输入文件：{src}")
        print(f"请把数据保存为 {DEFAULT_CSV.name}，放在 {HERE}")
        print("格式：两列，第一行表头 raman_shift_cm-1,intensity")
        return 1

    x, y = load_csv(src)
    if y.size < 12:
        print(f"只有 {y.size} 个点，无法可靠拟合。至少需要 12 个点，建议 300 点以上。")
        return 1

    order, note = auto_order(y.size, args.order)
    if note:
        print("[提示]", note)

    F, R = split_channels(y, args.method, order, x=x)
    m = qc_metrics(x, y, F, R, args.method, order, 1e6, 0.01, 25)
    state, issue = verdict(m)

    np.savetxt(OUT_CSV, np.column_stack([x, y, F, R]), delimiter=",", fmt="%.6f",
               comments="",
               header="raman_shift,intensity,fluorescence,raman")
    plot_channels(x, y, F, R, OUT_PNG,
                  f"my_ramantest（{args.method}, order={order}）",
                  m, close=args.no_show)

    print("=" * 60)
    print(f"输入      : {src}")
    print(f"点数      : {x.size}    位移范围: {x.min():.0f}–{x.max():.0f} cm-1")
    print(f"方法      : {args.method}   阶数: {order}")
    print(f"荧光占比  : {F.mean() / y.mean():.1%}（荧光估计均值 / 原谱均值）")
    print("-" * 60)
    print(f"质检结论  : {state}" + (f"（{issue}）" if issue else ""))

    def fmt(v, nd=2, unit=""):
        return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}{unit}"

    print(f"  负向过冲     {fmt(m['负向过冲'], 3)}   <0.25 为佳")
    print(f"  峰位最大偏移  {fmt(m['峰位最大偏移'], 1, ' cm-1')}   <5 为佳")
    print(f"  峰面积保留率  {fmt(m['峰面积保留率'], 2)}   0.8–1.2 为佳")
    print(f"  检出峰数     {m['峰数']}")
    print(f"  粗糙度比     {fmt(m['粗糙度比'], 4)}   <0.5 为佳（荧光应远比拉曼平滑）")
    print("=" * 60)
    print("通道数据  :", OUT_CSV)
    print("结果图    :", OUT_PNG)

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\n图窗已打开（关闭后程序结束）；不弹窗请加 --no-show")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
