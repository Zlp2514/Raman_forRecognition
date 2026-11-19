"""P13：结果的稳定性——把"划分运气"和"批次难度"分开。

为什么要做
    前面所有实验都只跑了一次随机划分。评审最容易提的两个问题是：
      · 你这个 0.932 是不是恰好抽到了一组好划分？
      · 留批次的标准差 0.066 说明什么？

    这两件事其实衡量的是不同的波动来源，必须分开报：

        随机划分的种子波动  = 划分运气（重抽训练/测试集，看准确率抖多少）
        留批次的折间波动    = 批次难度（不同批次本身有难有易）

    本脚本对三种配置各跑 5 个种子，给出"均值 ± 种子标准差"，
    再把留批次的折间标准差并列，两者含义不同，不可混为一谈。

用法
    .venv/Scripts/python.exe 16_seed_stability.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil
from raman_rec.classify import group_cv_eval, random_split_eval

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SEEDS = (0, 1, 2, 3, 4)

CONFIGS = {
    "全维（推荐）": {"n_components": None, "band": None},
    "PCA=20（旧基线）": {"n_components": 20, "band": None},
    "指纹区": {"n_components": None, "band": (500.0, 1800.0)},
}


def subset_band(ds, band):
    """只保留指定波数范围（返回副本，不修改原数据集）。"""
    import copy

    lo, hi = band
    out = copy.copy(ds)
    out.spectra = [s.slice(lo, hi) for s in ds.spectra]
    return out


def run(ds, name: str) -> dict:
    """每个配置同时给出：随机划分的种子波动 + 留批次的折间波动。"""
    import copy

    out = {}
    for cfg_name, cfg in CONFIGS.items():
        target = subset_band(ds, cfg["band"]) if cfg["band"] else copy.copy(ds)
        accs = np.array([
            random_split_eval(target, test_size=0.3, seed=seed,
                              n_components=cfg["n_components"])["accuracy"]
            for seed in SEEDS
        ])
        g = group_cv_eval(target, group_key="batch",
                          n_components=cfg["n_components"])
        out[cfg_name] = {
            "seed_mean": float(accs.mean()),
            "seed_std": float(accs.std(ddof=1)),
            "seed_min": float(accs.min()),
            "seed_max": float(accs.max()),
            "fold_mean": float(g["mean_accuracy"]),
            "fold_std": float(g["std_accuracy"]),
            "n_folds": len(g["folds"]),
        }
        r = out[cfg_name]
        print(f"  {cfg_name:16s} 随机 {r['seed_mean']:.3f} ± {r['seed_std']:.3f}"
              f"（{r['seed_min']:.3f}–{r['seed_max']:.3f}）"
              f" | 留批次 {r['fold_mean']:.3f} ± {r['fold_std']:.3f}"
              f"（{r['n_folds']} 折）")
    return out


def plot(results: dict[str, dict], out_path: Path) -> None:
    names = list(results)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(names))
    w = 0.34

    seed_means = [results[n]["seed_mean"] for n in names]
    seed_stds = [results[n]["seed_std"] for n in names]
    fold_means = [results[n]["fold_mean"] for n in names]
    fold_stds = [results[n]["fold_std"] for n in names]

    ax.bar(x - w / 2, seed_means, w, yerr=seed_stds, capsize=3,
           label="随机划分（5 个种子）", color="#54A24B",
           error_kw={"lw": 1, "ecolor": "#333333"})
    ax.bar(x + w / 2, fold_means, w, yerr=fold_stds, capsize=3,
           label="留一批次（折间）", color="#4C78A8",
           error_kw={"lw": 1, "ecolor": "#333333"})
    for xi, n in enumerate(names):
        ax.text(xi - w / 2, seed_means[xi] + seed_stds[xi] + 0.02,
                f"{seed_means[xi]:.3f}", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, fold_means[xi] + fold_stds[xi] + 0.02,
                f"{fold_means[xi]:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("准确率")
    ax.set_title("两种波动来源：划分运气 vs 批次难度", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="种子稳定性分析")
    parser.add_argument("--dataset", choices=["bacterial", "oil"], default="bacterial")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    ds = (load_bacterial(DATA_DIR / "Bacterial_data") if args.dataset == "bacterial"
          else load_oil(DATA_DIR / "Oil_data"))
    print(f"数据集：{args.dataset}（{len(ds)} 条，{len(ds.classes)} 类）")
    results = run(ds, args.dataset)

    plot(results, ROOT / f"seed_stability_{args.dataset}.png")
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ""

    lines = ["config,seed_mean,seed_std,seed_min,seed_max,fold_mean,fold_std"]
    for name, r in results.items():
        lines.append(",".join([name] + [
            fmt(r.get(k)) for k in
            ("seed_mean", "seed_std", "seed_min", "seed_max",
             "fold_mean", "fold_std")]))
    out = ROOT / f"seed_stability_{args.dataset}.csv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print("表已保存:", out)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
