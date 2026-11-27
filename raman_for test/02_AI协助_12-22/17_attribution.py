"""P14：误差的波数归因——模型到底靠哪几段谱在判断。

为什么需要
    前面的实验只知道"哪两类会混"（E-coli 与 R-terrigena 互相误判 31 与 15 条），
    但不知道"为什么混"。回答这个问题有两个互补的角度：

        判别权重   线性 SVM 的系数——模型实际使用了哪些波数
        波段重要性 把某个波段的值打乱后准确率掉多少——该波段对最终结果有多关键

    前者说明"模型怎么想"，后者说明"数据本身有没有信息"。两者结合，
    才能判断错误是"没有信息"还是"模型没用好信息"。

方法
    1. 在两类之间训练二分类线性 SVM，取系数曲线，与两类平均谱之差对照；
    2. 把全谱切成若干等宽波段，逐段打乱（保持其他波段不变），测准确率下降；
    3. 把系数绝对值最大的若干个波数聚成峰，输出峰位清单。

用法
    .venv/Scripts/python.exe 17_attribution.py
    .venv/Scripts/python.exe 17_attribution.py --pair E-coli R-terrigena
"""
from __future__ import annotations

import sys
from pathlib import Path

# 脚本已迁至 raman_for test/ 子目录：把仓库根目录加入模块搜索路径，
# 否则 from raman_rec import ... 会失败。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial
from raman_rec.classify import prepare_matrix
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def top_peaks(weights: np.ndarray, x: np.ndarray, k: int = 8,
              min_gap: float = 25.0) -> list[tuple[float, float]]:
    """取 |权重| 最大的若干波数，并按最小间距合并成峰（避免同一峰峰顶重复计数）。"""
    order = np.argsort(-np.abs(weights))
    picked: list[tuple[float, float]] = []
    for idx in order:
        shift = float(x[idx])
        if all(abs(shift - p[0]) >= min_gap for p in picked):
            picked.append((shift, float(weights[idx])))
        if len(picked) >= k:
            break
    return sorted(picked)


def band_importance(model, X, y, x, n_bands: int = 24, repeats: int = 3,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """逐波段打乱，返回每个波段的准确率下降与中心波数。"""
    rng = np.random.default_rng(seed)
    base = float(np.mean(model.predict(X) == y))
    d = X.shape[1]
    edges = np.linspace(0, d, n_bands + 1).astype(int)
    drops = np.zeros(n_bands)
    centers = np.zeros(n_bands)

    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        accs = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, lo:hi] = X[perm][:, lo:hi]
            accs.append(float(np.mean(model.predict(Xp) == y)))
        drops[b] = base - float(np.mean(accs))
        centers[b] = float(x[(lo + hi) // 2])
    return drops, centers


def main() -> int:
    parser = argparse.ArgumentParser(description="波数归因分析")
    parser.add_argument("--pair", nargs=2, default=["E-coli", "R-terrigena"])
    parser.add_argument("--n-bands", type=int, default=24)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    from sklearn.svm import SVC

    ds = load_bacterial(DATA_DIR / "Bacterial_data")
    X, y, _ = prepare_matrix(ds, normalize=None)
    X = row_snv(X)
    x = ds.spectra[0].x
    print(f"数据集：{len(ds)} 条，{len(ds.classes)} 类，{X.shape[1]} 个波数点")

    # ---------- 1) 两类之间的判别权重 ----------
    a, b = args.pair
    mask = (y == a) | (y == b)
    clf = SVC(kernel="linear", C=1.0).fit(X[mask], y[mask])
    coef = clf.coef_.ravel()
    if clf.classes_[0] != a:            # 保证权重正方向指向 a
        coef = -coef

    mu_a, mu_b = X[y == a].mean(axis=0), X[y == b].mean(axis=0)
    diff = mu_a - mu_b
    peaks = top_peaks(coef, x, k=8)
    acc_pair = float(np.mean(clf.predict(X[mask]) == y[mask]))
    print(f"两类判别（{a} vs {b}）：训练集准确率 {acc_pair:.3f}")
    print("判别权重最大的峰位（cm⁻¹）："
          + "、".join(f"{p:.0f}" for p, _ in sorted(peaks, key=lambda t: -abs(t[1]))))

    # ---------- 2) 全类别的波段重要性 ----------
    multi = SVC(kernel="linear", C=1.0).fit(X, y)
    drops, centers = band_importance(multi, X, y, x, n_bands=args.n_bands)
    order = np.argsort(-drops)[:6]
    print("最关键的波段（打乱后准确率下降）：")
    for i in order:
        print(f"  {centers[i]:7.0f} cm⁻¹  -{drops[i]*100:5.1f} pt")

    # ---------- 3) 作图 ----------
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2),
                             gridspec_kw={"height_ratios": [1.35, 1]})

    ax = axes[0]
    def smooth(v, k=17):
        """移动平均，只为显示可读；峰位仍取自未平滑的原始权重。"""
        kernel = np.ones(k) / k
        return np.convolve(v, kernel, mode="same")

    ax.plot(x, smooth(diff) / (np.abs(diff).max() + 1e-12), lw=1.3,
            color="#B0AFA8", label=f"平均谱差（{a} − {b}）")
    ax.plot(x, smooth(coef) / (np.abs(coef).max() + 1e-12), lw=1.6,
            color="#4C78A8", label="SVM 判别权重（已平滑显示）")
    for i, (shift, w) in enumerate(sorted(peaks)):
        y_text = 1.04 if i % 2 == 0 else 0.90
        ax.axvline(shift, color="#E45756", ls=":", lw=1.0, alpha=0.7)
        ax.text(shift, y_text, f"{shift:.0f}", rotation=90, fontsize=7.5,
                ha="right", va="bottom", color="#B23A48")
    pad = (x[-1] - x[0]) * 0.02
    ax.set_xlim(x[0] - pad, x[-1] + pad)
    ax.set_ylim(-1.15, 1.20)
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("归一化幅度")
    ax.set_title(f"两类判别：谱差与 SVM 权重（虚线为权重最大的 8 个峰位）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.bar(centers, drops * 100, width=(x[-1] - x[0]) / args.n_bands * 0.9,
           color="#54A24B")
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("准确率下降（百分点）")
    ax.set_title("波段重要性：打乱该波段后全类别准确率的下降", fontsize=10.5)
    ax.set_xlim(x[0], x[-1])
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    out = ROOT / "attribution_result.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("图已保存:", out)

    lines = ["type,position_cm-1,value"]
    for shift, w in peaks:
        lines.append(f"discriminative_peak,{shift:.1f},{w:.4f}")
    for i in np.argsort(-drops)[:10]:
        lines.append(f"band_importance,{centers[i]:.1f},{drops[i]:.4f}")
    (ROOT / "attribution_result.csv").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8-sig")
    print("表已保存:", ROOT / "attribution_result.csv")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
