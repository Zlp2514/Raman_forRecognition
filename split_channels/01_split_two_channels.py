"""把一条真实拉曼光谱拆成「荧光通道」与「拉曼通道」两条曲线。

产出（每条谱）
    <名字>_channels.csv   四列：raman_shift, intensity, fluorescence, raman
    <名字>_channels.png   三联图：原谱 / 荧光通道 / 拉曼通道

同时给出五项无真值质检（不需要标准答案也能算）：
    负值比例      拉曼通道 < 0 的采样点占比
    峰位最大偏移  拉曼通道的峰与"原谱去线性趋势"的峰，配对后的最大位移
    峰面积保留率  峰窗内 拉曼积分 / 原谱去弦积分
    阶数敏感度    仅多项式法：order±1 时荧光曲线的最大相对变化
    粗糙度比      F 的二阶差分 RMS / R 的二阶差分 RMS（越小说明 F 越平滑）

用法
    python split_two_channels.py --demo
    python split_two_channels.py --input demo --method imodpoly --order 5
    python split_two_channels.py --input demo/spectrum_01.csv --method asls
    python split_two_channels.py --input demo --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
DEMO = ROOT / "demo"

METHODS = ("imodpoly", "modpoly", "asls", "airpls", "snip", "lowpass")


# ----------------------------------------------------------------- 数据读写
def load_csv(path: Path):
    """两列 CSV：位移, 强度；第一行表头。"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path.name}: 需要两列（位移, 强度）")
    x, y = data[:, 0].astype(float), data[:, 1].astype(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{path.name}: 位移必须严格递增且不重复")
    return x, y


def setup_fonts():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------- 分离
def split_channels(y, method="imodpoly", order=5, lam=1e6, p=0.01,
                   sigma=25, x=None):
    """返回 (荧光 F, 拉曼 R)。R = y - F 恒成立。"""
    if method == "lowpass":
        F = gaussian_filter1d(y.astype(float), sigma=sigma, mode="nearest")
        return F, y - F

    if x is None:
        x = np.arange(y.size, dtype=float)
    from pybaselines import Baseline
    fitter = Baseline(x_data=x)

    if method in ("imodpoly", "modpoly"):
        F, _ = getattr(fitter, method)(y, poly_order=order)
    elif method == "asls":
        F, _ = fitter.asls(y, lam=lam, p=p)
    elif method == "airpls":
        F, _ = fitter.airpls(y, lam=lam)
    elif method == "snip":
        F, _ = fitter.snip(y, max_half_window=order * 8)
    else:
        raise ValueError(f"未知方法：{method}")
    return np.asarray(F, float), y - F


# ----------------------------------------------------------------- 质检
def qc_metrics(x, y, F, R, method, order, lam, p, sigma):
    m = {}
    span = float(np.ptp(y)) or 1.0

    # 负向过冲：扣完荧光后谱是零均值残差，"一半点为负"是正常的，
    # 真正该看的是最深的负值相对主峰有多深。
    m["负值点占比"] = float(np.mean(R < 0))
    m["负向过冲"] = float(abs(min(R.min(), 0.0)) / (abs(R.max()) + 1e-12))

    # 参考峰：原谱去掉线性趋势，轻度平滑后再找峰（避免在噪声上取峰）
    trend = np.polyval(np.polyfit(x, y, 1), x)
    y_det = gaussian_filter1d(y - trend, 2.0, mode="nearest")
    pk_r, pr = find_peaks(R, prominence=0.06 * span)
    pk_y, _ = find_peaks(y_det, prominence=0.06 * span)

    # 只保留最显著的若干个峰，且只在 ±30 cm⁻¹ 内配对，避免跨峰误配
    keep = np.argsort(pr["prominences"])[::-1][:6]
    pk_r = pk_r[np.sort(keep)]
    dx_all = float(np.median(np.diff(x)))
    tol = max(4, int(round(30.0 / dx_all)))

    # 峰窗按「波数」取，不能按固定点数——否则采样间隔一变，窗口宽度就跟着变
    dx = float(np.median(np.diff(x)))
    half = max(6, int(round(40.0 / dx)))     # ±40 cm⁻¹

    shifts, ratios = [], []
    for i in pk_r:
        if pk_y.size == 0:
            continue
        nearest = pk_y[np.argmin(np.abs(pk_y - i))]
        if abs(int(nearest) - int(i)) > tol:
            continue
        shifts.append(abs(float(x[nearest] - x[i])))

        lo, hi = max(i - half, 0), min(i + half, y.size - 1)
        if hi - lo < 5:
            continue
        xs = x[lo:hi + 1]
        chord = np.interp(xs, [x[lo], x[hi]], [y[lo], y[hi]])
        area_ref = trapezoid(y[lo:hi + 1] - chord, xs)
        area_r = trapezoid(R[lo:hi + 1], xs)
        if abs(area_ref) > 1e-9:
            ratios.append(area_r / area_ref)

    m["峰位最大偏移"] = float(np.max(shifts)) if shifts else float("nan")
    m["峰面积保留率"] = float(np.mean(ratios[:3])) if ratios else float("nan")
    m["峰数"] = int(pk_r.size)

    d2F = np.diff(F, 2)
    d2R = np.diff(R, 2)
    m["粗糙度比"] = float(np.sqrt(np.mean(d2F ** 2)) /
                        (np.sqrt(np.mean(d2R ** 2)) + 1e-12))

    if method in ("imodpoly", "modpoly"):
        devs = []
        for o in (order - 1, order + 1):
            if o < 1:
                continue
            F2, _ = split_channels(y, method, o, lam, p, sigma, x)
            devs.append(float(np.max(np.abs(F2 - F)) / span))
        m["阶数敏感度"] = max(devs) if devs else float("nan")
    else:
        m["阶数敏感度"] = float("nan")
    return m


def verdict(m):
    issues = []
    if m["负向过冲"] > 0.25:
        issues.append("负向过冲")
    if not np.isnan(m["峰面积保留率"]) and not (0.8 <= m["峰面积保留率"] <= 1.2):
        issues.append("峰面积异常")
    if not np.isnan(m["峰位最大偏移"]) and m["峰位最大偏移"] > 5:
        issues.append("峰位偏移")
    if not np.isnan(m["阶数敏感度"]) and m["阶数敏感度"] > 0.05:
        issues.append("阶数敏感")
    if m["粗糙度比"] > 0.5:
        issues.append("荧光不平滑")
    return ("通过" if not issues else "警告"), "；".join(issues)


# ----------------------------------------------------------------- 绘图
def plot_channels(x, y, F, R, path, title, m, close=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.8), sharex=True)

    axes[0].plot(x, y, color="0.55", lw=0.8, label="原谱")
    axes[0].plot(x, F, color="tab:orange", lw=1.3, label="估计荧光")
    axes[0].set_title(f"{title}｜原谱与估计荧光", fontsize=11)
    axes[0].set_ylabel("强度")
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=0.92)

    axes[1].plot(x, F, color="tab:orange", lw=1.4)
    axes[1].fill_between(x, F.min(), F, color="tab:orange", alpha=0.12)
    axes[1].set_title("荧光通道 F(ν)", fontsize=11)
    axes[1].set_ylabel("荧光强度")

    axes[2].plot(x, R, color="tab:blue", lw=0.8)
    axes[2].axhline(0, color="0.4", lw=0.6, ls=":")
    axes[2].set_title("拉曼通道 R(ν) = 原谱 − 荧光", fontsize=11)
    axes[2].set_ylabel("拉曼强度")
    axes[2].set_xlabel(r"拉曼位移 (cm$^{-1}$)")

    qc_line = ("过冲 %.2f　·　峰位最大偏移 %.1f cm$^{-1}$　·　"
               "峰面积保留 %.2f　·　峰数 %d"
               % (m["负向过冲"], m["峰位最大偏移"],
                  m["峰面积保留率"], m["峰数"]))

    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.012, qc_line, ha="center", va="bottom",
             fontsize=9, color="0.4")
    fig.savefig(path, dpi=200)
    if close:
        plt.close(fig)


def plot_compare(x, results, path, close=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    for (name, (F, R)), c in zip(results.items(), colors):
        axes[0].plot(x, F, lw=1.3, color=c, label=name)
        axes[1].plot(x, R, lw=0.8, color=c, label=name, alpha=0.85)

    axes[0].set_title("不同方法给出的荧光通道", fontsize=11)
    axes[0].set_ylabel("F(ν)")
    axes[1].set_title("对应的拉曼通道", fontsize=11)
    axes[1].set_ylabel("R(ν)")
    axes[1].set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if close:
        plt.close(fig)


# ----------------------------------------------------------------- 示例数据
def make_demo(force=False):
    """生成 3 条合成谱：荧光形状不同、峰强不同，用于演示批量与质检。

    已存在的文件默认不重写——避免被 Excel / WPS / VS Code 预览占用时抛出
    PermissionError。要重新生成请先删除 demo 目录或加 --force。
    """
    DEMO.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261127)
    x = np.linspace(200, 2000, 900)

    def g(c, h, w):
        return h * np.exp(-0.5 * ((x - c) / w) ** 2)

    peaks = (g(520, 150, 10) + g(1000, 220, 14)
             + g(1450, 130, 11) + g(1650, 180, 16))

    specs = [
        ("spectrum_01", 300 + 900 * np.exp(-(x - 200) / 700) + 0.00035 * (x - 1000) ** 2),
        ("spectrum_02", 500 + 1400 * np.exp(-(x - 200) / 400)),
        ("spectrum_03", 200 + 600 * np.exp(-(x - 200) / 1100) + 0.0009 * (x - 200) ** 2),
    ]
    paths = []
    for name, bg in specs:
        y = bg + peaks + rng.normal(0, 4, x.size)
        p = DEMO / f"{name}.csv"
        if p.exists() and not force:
            paths.append(p)
            continue
        try:
            np.savetxt(p, np.column_stack([x, y]), delimiter=",", fmt="%.4f",
                       comments="", header="raman_shift_cm-1,intensity")
        except PermissionError:
            print(f"[警告] {p.name} 正被其他程序占用（Excel / WPS / 编辑器预览），"
                  f"跳过重写，沿用已有文件")
        paths.append(p)
    return paths


# ----------------------------------------------------------------- 主流程
def process(paths, args, outdir):
    rows = []
    for k, path in enumerate(paths, 1):
        try:
            x, y = load_csv(path)
        except Exception as exc:
            print(f"[跳过] {path.name}: {exc}")
            continue

        F, R = split_channels(y, args.method, args.order, args.lam, args.p,
                              args.sigma, x)
        m = qc_metrics(x, y, F, R, args.method, args.order, args.lam, args.p,
                       args.sigma)
        state, issue = verdict(m)

        out_csv = outdir / f"{path.stem}_channels.csv"
        out_png = outdir / f"{path.stem}_channels.png"
        locked = False
        try:
            np.savetxt(out_csv, np.column_stack([x, y, F, R]), delimiter=",",
                       fmt="%.6f", comments="",
                       header="raman_shift,intensity,fluorescence,raman")
            plot_channels(x, y, F, R, out_png,
                          f"{path.stem}（{args.method}, "
                          f"{'order=' + str(args.order) if args.method in ('imodpoly', 'modpoly') else '默认参数'}）",
                          m, close=args.no_show)
        except PermissionError:
            locked = True
            state, issue = "未写出", "文件被占用"
            print(f"      ↳ {out_csv.name} / {out_png.name} 被其他程序占用，"
                  f"本次未写出（关闭 Excel / WPS / 预览后重试）")

        rows.append(dict(file=path.name, method=args.method, state=state,
                         issue=issue, **m))
        print(f"[{k}/{len(paths)}] {path.name:22s} {state}  "
              f"过冲{m['负向过冲']:5.2f} 面积保留{m['峰面积保留率']:5.2f} "
              f"偏移{m['峰位最大偏移']:4.1f} 粗糙比{m['粗糙度比']:5.3f}  {issue}")

    if not rows:
        return None

    keys = ["file", "method", "state", "issue", "负向过冲", "负值点占比",
            "峰位最大偏移", "峰面积保留率", "峰数", "粗糙度比", "阶数敏感度"]
    table = outdir / "qc_summary.csv"
    try:
        with table.open("w", encoding="utf-8-sig") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(
                    (f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]))
                    for k in keys) + "\n")
    except PermissionError:
        print(f"[警告] {table.name} 被占用，质检表未更新；请关闭后重试")
    return table, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="拆出荧光通道与拉曼通道")
    ap.add_argument("--input", type=Path, help="单个 CSV 或目录")
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--method", default="imodpoly", choices=METHODS)
    ap.add_argument("--order", type=int, default=5, help="多项式阶数 / snip 窗口系数")
    ap.add_argument("--lam", type=float, default=1e6)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=25, help="lowpass 的平滑尺度")
    ap.add_argument("--demo", action="store_true", help="生成 3 条示例谱并处理")
    ap.add_argument("--force", action="store_true", help="强制重写已存在的示例谱")
    ap.add_argument("--compare", action="store_true", help="对比四种方法的荧光通道")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--limit", type=int, default=50, help="批量最多处理多少个文件")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    setup_fonts()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.input is None:
        if not args.demo:
            print("未指定 --input，默认处理内置示例（3 条合成谱）。")
            print("要处理自己的数据：--input 你的文件.csv")
            print("只看用法：--help\n")
        made = make_demo(force=args.force)
        args.input = DEMO
        print(f"示例谱 {len(made)} 条 -> {DEMO}")

    if args.input.is_dir():
        files = sorted(p for p in args.input.glob("*.csv")
                       if not p.stem.endswith("_channels"))[:args.limit]
    else:
        files = [args.input]

    print(f"方法 {args.method}｜参数 order={args.order} lam={args.lam:g} "
          f"p={args.p} sigma={args.sigma}｜共 {len(files)} 条\n")

    res = process(files, args, args.outdir)
    if res is None:
        print("没有可处理的文件。")
        return 1
    table, rows = res

    if args.compare and files:
        x, y = load_csv(files[0])
        variants = {
            "imodpoly(5)": split_channels(y, "imodpoly", 5, x=x),
            "asls": split_channels(y, "asls", x=x),
            "airpls": split_channels(y, "airpls", x=x),
            "lowpass(25)": split_channels(y, "lowpass", sigma=25, x=x),
        }
        p = args.outdir / f"compare_{files[0].stem}.png"
        plot_compare(x, variants, p, close=args.no_show)
        print(f"\n对比图: {p}")

    ok = sum(r["state"] == "通过" for r in rows)
    print(f"\n汇总: {ok}/{len(rows)} 条通过质检")
    print(f"明细: {table}")
    print(f"图与通道数据: {args.outdir}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\n图窗已打开（关闭后程序结束）；不想要弹窗请加 --no-show")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
