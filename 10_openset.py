"""P7：开放集识别——模型能否识别出"没见过的东西"。

任务构造（留一类法）
    每次从 N 个类别中挑 1 类作为"未知类"，完全不参与训练；
    模型只用其余 N-1 类训练，测试时把已知类与未知类混在一起，
    要求它既要判对已知类，也要把未知类拒掉。

两种协议
    random    已知类的 70/30 分层划分；未知类全部作为测试样本
    loo-batch 留一个完整批次做测试（已知类与未知类都在其中），
              训练集来自其余批次的已知类——这是真实部署的组合

三种置信度
    margin       SVM 决策值（离所有边界都远 = 不像已知）
    prototype    到最近类原型的欧氏距离
    mahalanobis  到最近类中心的马氏距离（考虑类内松散程度，通常最可靠）

核心指标
    AUROC 不依赖阈值，最适合比较评分器；
    open_acc_all 是含拒识代价的真实准确率（未知判对也算对）。

用法
    .venv/Scripts/python.exe 10_openset.py
    .venv/Scripts/python.exe 10_openset.py --dataset oil --keep-rate 0.9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.openset import OpenSetModel, evaluate_openset
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCORERS = ("margin", "prototype", "mahalanobis")
SCORER_CN = {"margin": "SVM 决策值", "prototype": "原型距离", "mahalanobis": "马氏距离"}


def split_fit_calib(idx: np.ndarray, seed: int, ratio: float = 0.15):
    """把训练索引分成"拟合"与"校准"两部分。

    校准集只用来确定拒识阈值，不参与模型拟合——若用拟合样本定阈值，
    会因训练样本天然靠近类中心而过度拒识。
    """
    rng = np.random.default_rng(seed)
    idx = np.array(idx, dtype=int)
    shuffled = rng.permutation(idx)
    n_calib = max(40, int(len(idx) * ratio))
    return shuffled[n_calib:], shuffled[:n_calib]


def run_random(ds: Dataset, X, y, groups, unknown_cls: str, keep_rate: float,
               seed: int = 0) -> dict:
    """随机协议：已知类 70/30 划分，未知类全部作测试。"""
    rng = np.random.default_rng(seed)
    known_idx = np.where(y != unknown_cls)[0]
    unknown_idx = np.where(y == unknown_cls)[0]

    tr, te = [], []
    for c in np.unique(y[known_idx]):
        idx = known_idx[y[known_idx] == c]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * 0.3)))
        te.extend(idx[:n_te])
        tr.extend(idx[n_te:])

    fit_idx, calib_idx = split_fit_calib(np.array(tr), seed + 1)
    model = OpenSetModel().fit(row_snv(X[fit_idx]), y[fit_idx])
    return evaluate_openset(model, row_snv(X[calib_idx]),
                            row_snv(X[np.array(te)]), y[np.array(te)],
                            row_snv(X[unknown_idx]), keep_rate)


def run_leave_batch(ds: Dataset, X, y, groups, unknown_cls: str,
                    keep_rate: float) -> dict:
    """严格协议：留一个批次，已知类与未知类都在测试集里。"""
    per_key: dict[str, list[dict]] = {k: [] for k in SCORERS}
    for g in sorted({g for g in groups if g}):
        te_idx = np.where(groups == g)[0]
        tr_idx = np.where((groups != g) & (groups != "") & (y != unknown_cls))[0]
        if tr_idx.size < 50 or te_idx.size < 10:
            continue
        known_te = te_idx[y[te_idx] != unknown_cls]
        unknown_te = te_idx[y[te_idx] == unknown_cls]
        if known_te.size < 5 or unknown_te.size < 3:
            continue

        fit_idx, calib_idx = split_fit_calib(tr_idx, hash(g) % 1000)
        model = OpenSetModel().fit(row_snv(X[fit_idx]), y[fit_idx])
        res = evaluate_openset(model, row_snv(X[calib_idx]),
                               row_snv(X[known_te]), y[known_te],
                               row_snv(X[unknown_te]), keep_rate)
        for k in SCORERS:
            per_key[k].append(res[k])

    out = {}
    for k in SCORERS:
        items = per_key[k]
        if not items:
            continue
        out[k] = {name: float(np.mean([it[name] for it in items]))
                  for name in ("auroc", "closed_acc", "open_acc",
                               "reject_rate", "open_acc_all", "known_accept")}
        out[k]["auroc_std"] = float(np.std([it["auroc"] for it in items]))
    return out


def evaluate_dataset(ds: Dataset, keep_rate: float) -> dict:
    X, y, groups = prepare_matrix(ds, normalize=None)
    groups = np.asarray(groups)
    classes = list(ds.classes)

    rand_acc: dict[str, list[dict]] = {k: [] for k in SCORERS}
    lb_acc: dict[str, list[dict]] = {k: [] for k in SCORERS}

    for cls in classes:
        r = run_random(ds, X, y, groups, cls, keep_rate)
        for k in SCORERS:
            rand_acc[k].append(r[k])
        b = run_leave_batch(ds, X, y, groups, cls, keep_rate)
        for k in SCORERS:
            if k in b:
                lb_acc[k].append(b[k])
        print(f"  未知类 {cls:14s} | "
              f"随机 AUROC 马氏 {r['mahalanobis']['auroc']:.3f}"
              f" | 留批次 马氏 {b.get('mahalanobis', {}).get('auroc', float('nan')):.3f}")

    summary = {}
    for proto, acc in (("random", rand_acc), ("loo_batch", lb_acc)):
        summary[proto] = {}
        for k in SCORERS:
            items = acc[k]
            if not items:
                continue
            summary[proto][k] = {
                name: float(np.mean([it[name] for it in items]))
                for name in ("auroc", "closed_acc", "open_acc",
                             "reject_rate", "open_acc_all")
            }
            summary[proto][k]["auroc_std"] = float(
                np.std([it["auroc"] for it in items]))
    return summary


def plot_summary(results: dict[str, dict], out_path: Path, keep_rate: float) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.0), squeeze=False)
    x = np.arange(len(SCORERS))
    width = 0.38

    for ax, (ds_name, s) in zip(axes[0], results.items()):
        rand = [s["random"][k]["auroc"] for k in SCORERS]
        loo = [s["loo_batch"][k]["auroc"] for k in SCORERS]
        ax.bar(x - width / 2, rand, width, label="随机划分", color="#54A24B")
        ax.bar(x + width / 2, loo, width, label="留一批次", color="#4C78A8",
               yerr=[s["loo_batch"][k]["auroc_std"] for k in SCORERS],
               capsize=3, error_kw={"lw": 1, "ecolor": "#333333"})
        ax.axhline(0.5, color="#888780", ls=":", lw=1.2)
        ax.text(-0.45, 0.505, "随机猜测 0.5", fontsize=8, ha="left",
                va="bottom", color="#5F5E5A")
        ax.set_xticks(x)
        ax.set_xticklabels([SCORER_CN[k] for k in SCORERS], fontsize=10)
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel("AUROC（已知 vs 未知）")
        ax.set_xlabel("误差棒 = 不同类别作为未知时的标准差", fontsize=8,
                      color="#5F5E5A")
        ax.set_title(f"{ds_name}：开放集评分器对比（保留已知 {keep_rate:.0%}）",
                     fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="开放集识别实验")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--keep-rate", type=float, default=0.95,
                        help="已知类的接受率，1-该值即允许的误拒比例")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    results = {}
    if args.dataset in ("bacterial", "both"):
        print("=" * 72)
        print("数据集：bacterial（6 类菌，留一类作未知）")
        results["bacterial"] = evaluate_dataset(
            load_bacterial(DATA_DIR / "Bacterial_data"), args.keep_rate)
    if args.dataset in ("oil", "both"):
        print("=" * 72)
        print("数据集：oil（8 类油，留一类作未知）")
        results["oil"] = evaluate_dataset(
            load_oil(DATA_DIR / "Oil_data"), args.keep_rate)

    print("=" * 72)
    for ds_name, s in results.items():
        for proto, label in (("random", "随机划分"), ("loo_batch", "留一批次")):
            print(f"[{ds_name} / {label}]")
            print(f"  {'评分器':14s}{'AUROC':>8s}{'闭集准确率':>12s}"
                  f"{'未知拒识率':>12s}{'开放集准确率':>14s}")
            for k in SCORERS:
                if k not in s[proto]:
                    continue
                m = s[proto][k]
                print(f"  {SCORER_CN[k]:14s}{m['auroc']:>8.3f}"
                      f"{m['closed_acc']:>12.3f}{m['reject_rate']:>12.3f}"
                      f"{m['open_acc_all']:>14.3f}")

    plot_summary(results, ROOT / "openset_result.png", args.keep_rate)

    lines = ["dataset,protocol,scorer,auroc,auroc_std,closed_acc,reject_rate,open_acc_all"]
    for ds_name, s in results.items():
        for proto in ("random", "loo_batch"):
            for k in SCORERS:
                if k not in s[proto]:
                    continue
                m = s[proto][k]
                lines.append(f"{ds_name},{proto},{k},{m['auroc']:.4f},"
                             f"{m.get('auroc_std', 0):.4f},{m['closed_acc']:.4f},"
                             f"{m['reject_rate']:.4f},{m['open_acc_all']:.4f}")
    (ROOT / "openset_result.csv").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8-sig")
    print("表已保存:", ROOT / "openset_result.csv")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
