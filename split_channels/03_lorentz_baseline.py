"""按王忠（2021）论文 3.4 节复现「洛伦兹峰 + 偏移」基线校准，并与 b(x) 对比。

论文思路（五步）
    ① 寻峰，得到每个峰的极大值点 c
    ② 从极大值点向左右找拐点，确定该峰的拟合区间
    ③ 令 t = (x - c)²，把 (x, y) 换成 (t, y)
    ④ 拟合 y = a / (t + b) + d，解出 a、b、d（d 即荧光造成的局部偏移）
    ⑤ 重建时令 d = 0，用 a、b、c 还原每个峰，再把所有峰叠加 → 去背景后的谱

与 pybaselines 的差别
    pybaselines 拟合「整条背景 b(x)」，R = y - b(x)，不依赖寻峰；
    本方法拟合「每个峰」，背景隐含在 d 里，强依赖寻峰是否准确。

用法
    python 03_lorentz_baseline.py                 # 洛伦兹峰合成谱（方法的主场）
    python 03_lorentz_baseline.py --shape gauss   # 高斯峰合成谱（看它何时失效）
    python 03_lorentz_baseline.py --csv my.csv    # 自己的两列数据（无真值）
    python 03_lorentz_baseline.py --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------- 合成数据
def make_demo(shape="lorentz", n=1600, seed=7):
    """造一条带荧光背景的合成谱，返回 x, y, 真实拉曼信号。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(200, 2000, n)

    centers = [520, 1000, 1450, 1650]
    heights = [260, 380, 220, 300]
    fwhm = [16, 22, 14, 20]
    signal = np.zeros_like(x)
    for c, h, w in zip(centers, heights, fwhm):
        if shape == "lorentz":
            b = (w / 2.0) ** 2
            signal += (h * b) / ((x - c) ** 2 + b)      # 峰顶 = h*b/b = h
        else:
            sigma = w / 2.3548
            signal += h * np.exp(-0.5 * ((x - c) / sigma) ** 2)

    bg = 300 + 900 * np.exp(-(x - 200) / 700) + 0.00035 * (x - 1000) ** 2
    y = signal + bg + rng.normal(0, 4, x.size)
    return x, y, signal


def load_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    if d.ndim != 2 or d.shape[1] < 2:
        raise ValueError("需要两列：位移, 强度")
    x, y = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(x)
    return x[o], y[o]


# --------------------------------------------------------------- 核心算法
def peak_windows(x, y, prominence_frac=0.05, min_half_cm=5.0, max_half_cm=150.0,
                 window_scale=2.0):
    """① 寻峰；② 用两侧斜率最大处（≈ 二阶导过零的拐点）定拟合区间。

    window_scale：在拐点基础上向外放宽的倍数。论文只取到拐点（=1.0），
    但洛伦兹峰的拐点离峰心仅约 ±0.5 FWHM，窗太窄会让 b 与 a 强相关、定不准，
    因此默认放宽到 2 倍，仍受相邻峰中点约束。
    """
    span = float(np.ptp(y)) or 1.0
    idx, props = find_peaks(y, prominence=prominence_frac * span)
    if idx.size == 0:
        return []

    ys = gaussian_filter1d(y, 2.0, mode="nearest")
    d1 = np.gradient(ys, x)

    dx = float(np.median(np.diff(x)))
    min_half = max(2, int(round(min_half_cm / dx)))
    max_half = max(min_half + 2, int(round(max_half_cm / dx)))

    peaks_idx = list(idx)
    windows = []
    for k, i in enumerate(peaks_idx):
        # 与相邻峰的中间点作为硬边界，避免互相侵占
        left_bound = peaks_idx[k - 1] if k > 0 else 0
        right_bound = peaks_idx[k + 1] if k < len(peaks_idx) - 1 else y.size - 1
        mid_l = (left_bound + i) // 2
        mid_r = (i + right_bound) // 2

        lo_lim = max(i - max_half, mid_l, 0)
        hi_lim = min(i + max_half, mid_r, y.size - 1)

        # 左侧 |斜率| 最大处为拐点
        seg_l = slice(max(lo_lim, i - max_half), i)
        if seg_l.stop - seg_l.start >= 3:
            lo = seg_l.start + int(np.argmax(np.abs(d1[seg_l])))
        else:
            lo = i - min_half
        seg_r = slice(i + 1, min(hi_lim + 1, y.size))
        if seg_r.stop - seg_r.start >= 3:
            hi = seg_r.start + int(np.argmax(np.abs(d1[seg_r])))
        else:
            hi = i + min_half

        # 按倍数放宽（论文原法 scale=1），但不超过相邻峰中点
        half_l = max(min_half, int(round((i - lo) * window_scale)))
        half_r = max(min_half, int(round((hi - i) * window_scale)))
        lo = max(i - half_l, mid_l, 0)
        hi = min(i + half_r, mid_r, y.size - 1)
        if hi - lo < 6:
            continue
        windows.append((lo, hi, i))
    return windows


def fit_one_peak(x, y, lo, hi, c_index):
    """③④ 令 t=(x-c)²；对固定 b 用线性最小二乘解 (a, d)，网格搜 b。"""
    xs, ys = x[lo:hi + 1], y[lo:hi + 1]
    c = x[c_index]
    t = (xs - c) ** 2

    best = None
    t_max = max(t.max(), 1e-6)
    for b in np.geomspace(t_max * 1e-4 + 1e-6, t_max * 4, 400):
        u = 1.0 / (t + b)
        A = np.column_stack([u, np.ones_like(u)])      # y = a*u + d
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - A @ coef
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, coef[0], b, coef[1])

    _, a, b, d = best
    return {"c": c, "a": a, "b": b, "d": d, "lo": lo, "hi": hi}


def reconstruct(x, params):
    """⑤ 令 d=0，用 a、b、c 还原每个峰并叠加。

    注意：拟合窗口只用来「定参」，重建要在全轴上做——洛伦兹峰的拐点离峰心
    只有约 ±0.5 FWHM，若只在窗内重建，峰尾会被截掉，窗外留下大块残差。
    """
    clean = np.zeros_like(x, dtype=float)
    for p in params:
        t = (x - p["c"]) ** 2
        clean += p["a"] / (t + p["b"])
    return clean


def lorentz_baseline(x, y, **kw):
    wins = peak_windows(x, y, **kw)
    params = [fit_one_peak(x, y, lo, hi, i) for lo, hi, i in wins]
    clean = reconstruct(x, params)
    return clean, y - clean, params


def poly_reference(x, y, order=5):
    from pybaselines import Baseline
    b, _ = Baseline(x_data=x).imodpoly(y, poly_order=order)
    return y - b, b


# --------------------------------------------------------------- 绘图
def plot(x, y, clean, implied, ref_clean, title, path, truth=None, show=True):
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)

    axes[0].plot(x, y, color="0.6", lw=0.8, label="原始谱")
    axes[0].plot(x, implied, color="tab:orange", lw=1.3, label="隐含背景 y − 峰和")
    axes[0].set_title(f"{title}｜原始谱与还原出的背景", fontsize=11)
    axes[0].set_ylabel("强度")
    axes[0].legend(fontsize=8.5, loc="upper right")

    if truth is not None:
        axes[1].plot(x, truth, color="k", ls="--", lw=1.4, label="真实拉曼信号")
    axes[1].plot(x, clean, color="tab:blue", lw=1.0, label="洛伦兹法重建")
    axes[1].plot(x, ref_clean, color="tab:green", lw=0.9, alpha=.85, label="IModPoly: y − b(x)")
    axes[1].axhline(0, color="0.4", lw=0.6, ls=":")
    axes[1].set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    axes[1].set_ylabel("去背景后强度")
    axes[1].set_title("两种方法的重建结果", fontsize=11)
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="复现洛伦兹峰基线校准（王忠 2021, 3.4 节）")
    ap.add_argument("--shape", default="lorentz", choices=("lorentz", "gauss"),
                    help="合成峰的线型；lorentz 是该方法的主场")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--order", type=int, default=5, help="对照用的 IModPoly 阶数")
    ap.add_argument("--prom", type=float, default=0.05, help="寻峰 prominence 阈值（占全谱幅度）")
    ap.add_argument("--window", type=float, default=2.0,
                    help="拟合窗口相对拐点的放大倍数；1.0 即论文原法")
    ap.add_argument("--sweep", action="store_true", help="扫描窗口倍数并打印对比表")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")

    if args.csv:
        x, y = load_csv(args.csv)
        truth = None
        title = args.csv.name
        tag = "csv"
    else:
        x, y, truth = make_demo(args.shape)
        title = f"合成谱（{'洛伦兹' if args.shape == 'lorentz' else '高斯'}峰 + 荧光背景）"
        tag = args.shape

    if args.sweep and truth is not None:
        ref_clean, _ = poly_reference(x, y, args.order)
        base = float(np.sqrt(np.mean((ref_clean - truth) ** 2)))
        print("=" * 64)
        print(f"窗口倍数扫描（{tag} 峰型）｜对照 IModPoly RMSE = {base:.2f}")
        print(f"{'scale':>6}{'峰数':>6}{'窗宽(cm-1)':>16}{'RMSE':>10}")
        for s in (1.0, 1.5, 2.0, 3.0, 4.0):
            wins = peak_windows(x, y, args.prom, window_scale=s)
            ps = [fit_one_peak(x, y, lo, hi, i) for lo, hi, i in wins]
            r = float(np.sqrt(np.mean((reconstruct(x, ps) - truth) ** 2)))
            w = ",".join(str(int(round(x[hi] - x[lo]))) for lo, hi, _ in wins)
            print(f"{s:>6.1f}{len(ps):>6}{w:>16}{r:>10.2f}")
        print("=" * 64)
        print("解读：scale=1.0 就是论文原文的「取到拐点」；窗太窄时 b 与 a 强相关，"
              "峰形定不准，RMSE 反而大。")
        return 0

    clean, implied, params = lorentz_baseline(x, y, prominence_frac=args.prom,
                                              window_scale=args.window)
    ref_clean, _ = poly_reference(x, y, args.order)

    print("=" * 64)
    print(f"数据      : {title}")
    print(f"采样      : {x.size} 点，{x.min():.0f}–{x.max():.0f} cm-1")
    print(f"窗口倍数  : {args.window}（1.0 = 论文原法）")
    print(f"检出峰数  : {len(params)}")
    for p in params:
        print(f"  峰心 {p['c']:7.1f} cm-1 ｜ a={p['a']:10.1f}  b={p['b']:7.1f}  "
              f"局部偏移 d={p['d']:8.1f}")
    if truth is not None:
        r1 = float(np.sqrt(np.mean((clean - truth) ** 2)))
        r2 = float(np.sqrt(np.mean((ref_clean - truth) ** 2)))
        print("-" * 64)
        print(f"相对真实信号的 RMSE：洛伦兹法 {r1:.3f} ｜ IModPoly {r2:.3f}")
        win = "洛伦兹法" if r1 < r2 else "IModPoly"
        print(f"这组数据上更接近真值：{win}（差 {abs(r1 - r2):.3f}）")

    out = HERE / f"lorentz_result_{tag}.png"
    plot(x, y, clean, implied, ref_clean, title, out, truth, show=not args.no_show)
    print("=" * 64)
    print("结果图    :", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
