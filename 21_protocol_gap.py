"""P18：论文图 1——随机划分与留一批次的差距。

这张图是全文的起点，要传达两件事：

    1. 随机划分系统性高估：同一模型同一数据，只换评估协议，准确率掉 10 个点左右；
    2. 掉点不是"运气"：五个随机种子的波动只有 ±1 pt，而逐批次的准确率分布在
       0.60–0.93 之间——批次本身有难有易。

为可复现，所有数字用项目管线（StandardScaler → SVC，无降维）现算：
    · 随机划分：5 个种子取均值 ± 标准差；
    · 留一批次：逐折准确率（含折名），均值 ± 折间标准差。

用法
    .venv/Scripts/python.exe 21_protocol_gap.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil, stratified_split
from raman_rec.datasets import group_split
from raman_rec.classify import prepare_matrix
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SEEDS = (0, 1, 2, 3, 4)
COLOR_RANDOM = "#54A24B"
COLOR_LOO = "#4C78A8"


def evaluate(ds, seed: int | None = None, folds=None):
    """用项目管线评估：seed 给定则做随机划分，否则按 folds 留一批次。"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    id2idx = {id(sp): i for i, sp in enumerate(ds.spectra)}
    X, y, _ = prepare_matrix(ds, normalize=None)
    X = row_snv(X)

    def fit_pred(tr, te):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="linear", C=1.0, random_state=0))])
        pipe.fit(X[tr], y[tr])
        return float(np.mean(pipe.predict(X[te]) == y[te]))

    if seed is not None:
        train, test = stratified_split(ds, test_size=0.3, seed=seed)
        tr = np.array([id2idx[id(s)] for s in train.spectra])
        te = np.array([id2idx[id(s)] for s in test.spectra])
        return fit_pred(tr, te), None

    accs, names = [], []
    for name, train, test in folds:
        tr = np.array([id2idx[id(s)] for s in train.spectra])
        te = np.array([id2idx[id(s)] for s in test.spectra])
        accs.append(fit_pred(tr, te))
        names.append(name)
    return float(np.mean(accs)), list(zip(names, accs))


def run(ds, name: str) -> dict:
    folds = group_split(ds, group_key="batch")
    seed_accs = np.array([evaluate(ds, seed=s)[0] for s in SEEDS])
    loo_mean, per_batch = evaluate(ds, folds=folds)
    loo_accs = np.array([a for _, a in per_batch])

    gap = seed_accs.mean() - loo_mean
    print(f"  {name}: 随机 {seed_accs.mean():.4f} ± {seed_accs.std(ddof=1):.4f}（5 种子）"
          f" | 留批次 {loo_mean:.4f} ± {loo_accs.std(ddof=1):.4f}（{len(loo_accs)} 折）"
          f" | 差距 {gap*100:.1f} pt")
    return {"random_mean": float(seed_accs.mean()),
            "random_std": float(seed_accs.std(ddof=1)),
            "loo_mean": loo_mean, "loo_std": float(loo_accs.std(ddof=1)),
            "gap_pt": gap * 100, "per_batch": per_batch}


def plot(results: dict[str, dict], out_path: Path) -> None:
    names = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # 左：两数据集 × 两协议
    x = np.arange(len(names))
    w = 0.34
    rm = [results[n]["random_mean"] for n in names]
    rs = [results[n]["random_std"] for n in names]
    lm = [results[n]["loo_mean"] for n in names]
    ls = [results[n]["loo_std"] for n in names]
    ax1.bar(x - w / 2, rm, w, yerr=rs, capsize=3, color=COLOR_RANDOM,
            label="随机划分（5 种子）", error_kw={"lw": 1, "ecolor": "#333333"})
    ax1.bar(x + w / 2, lm, w, yerr=ls, capsize=3, color=COLOR_LOO,
            label="留一批次（折间）", error_kw={"lw": 1, "ecolor": "#333333"})
    for xi, n in enumerate(names):
        ax1.text(xi - w / 2, rm[xi] + rs[xi] + 0.02, f"{rm[xi]:.3f}",
                 ha="center", fontsize=8.5)
        ax1.text(xi + w / 2, lm[xi] + ls[xi] + 0.02, f"{lm[xi]:.3f}",
                 ha="center", fontsize=8.5)
        gap = results[n]["gap_pt"]
        ax1.annotate(f"−{gap:.1f} pt", xy=(xi, 0.32), ha="center", fontsize=9.5,
                     color="#B23A48", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylim(0, 1.22)
    ax1.set_ylabel("准确率")
    ax1.set_title("同一模型，只换评估协议", fontsize=11)
    ax1.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
    ax1.grid(alpha=0.25, axis="y")

    # 右：逐批次准确率（批次有难有易）；两个数据集分段摆放，避免标签互相碰撞
    GAP = 1.0
    x_cursor = 0.0
    for i, n in enumerate(names):
        per = sorted(results[n]["per_batch"], key=lambda t: t[1])
        xs = x_cursor + np.arange(len(per))
        vals = [a for _, a in per]
        color = COLOR_RANDOM if i == 0 else "#E45790"
        ax2.scatter(xs, vals, s=46, color=color, alpha=0.9, zorder=3)
        for j, (g, a) in enumerate(per):
            ax2.annotate(g.replace("_Batch", "").replace("batch_", "B"),
                         (xs[j], a), textcoords="offset points",
                         xytext=(0, -14), fontsize=6.8,
                         ha="center", color="#5F5E5A")
        mean = results[n]["loo_mean"]
        ax2.hlines(mean, xs[0] - 0.3, xs[-1] + 0.3, colors=color,
                   ls=":", lw=1.6, alpha=0.9, zorder=2)
        ax2.text((xs[0] + xs[-1]) / 2, 0.565,
                 f"{n}：均值 {mean:.3f}", ha="center", fontsize=8.5, color=color)
        x_cursor = xs[-1] + GAP
    ax2.axvline(x_cursor - GAP / 2, color="#BBBBBB", lw=1.0)
    ax2.set_xlim(-0.6, x_cursor - GAP / 2 + 0.4)
    ax2.set_xlabel("批次（各数据集内按准确率排序）")
    ax2.set_ylabel("留一批次准确率")
    ax2.set_ylim(0.55, 1.03)
    ax2.set_title("逐批次准确率：批次本身有难有易", fontsize=11)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="双协议差距")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"], default="both")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    names = ["bacterial", "oil"] if args.dataset == "both" else [args.dataset]
    results: dict[str, dict] = {}
    lines = ["dataset,random_mean,random_std,loo_mean,loo_std,gap_pt"]
    for name in names:
        ds = (load_bacterial(DATA_DIR / "Bacterial_data") if name == "bacterial"
              else load_oil(DATA_DIR / "Oil_data"))
        print(f"数据集：{name}（{len(ds)} 条，{len(ds.classes)} 类）")
        results[name] = run(ds, name)
        r = results[name]
        lines.append(f"{name},{r['random_mean']:.4f},{r['random_std']:.4f},"
                     f"{r['loo_mean']:.4f},{r['loo_std']:.4f},{r['gap_pt']:.2f}")

    plot(results, ROOT / "protocol_gap.png")
    (ROOT / "protocol_gap.csv").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8-sig")
    print("表已保存:", ROOT / "protocol_gap.csv")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
