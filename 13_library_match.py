"""P11：谱库匹配的可行性——建库 + 比对 + 拒识，能到多少。

问题
    实际做分子识别，常见做法不是训练分类器，而是建一个谱库（每种物质存几条代表谱），
    待测谱进来与库比对，取最相似的一条；相似度太低就报"库中没有"。
    这条路线可行吗？能到多少准确率？库要建多大？

设计
    留一类法：每次把 1 类整体排除在库外当作"待测的未知物"，库只放其余类别。
    查询时已知类与未知类混在一起，要求既判对已知类、又拒掉未知类。

    三个量：闭集准确率（只算已知类判对）、库外拒识率（未知类被拒的比例）、
    开放集准确率（两者加权，含拒识代价）。

用法
    .venv/Scripts/python.exe 13_library_match.py
    .venv/Scripts/python.exe 13_library_match.py --dataset oil
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil, stratified_split
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.library import build_library, genuine_threshold, match
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

LIB_SIZES = (1, 3, 5, 10)


def evaluate_once(X_lib, y_lib, X_known, y_known, X_unknown, per_class, seed=0):
    """建库 → 定阈值 → 匹配 → 统计三项指标。"""
    from sklearn.metrics import roc_auc_score

    lib_X, lib_y = build_library(X_lib, y_lib, per_class=per_class, seed=seed)
    tau = genuine_threshold(X_lib, y_lib, lib_X, lib_y)

    s_known, p_known = match(X_known, lib_X, lib_y)
    s_unknown, _ = match(X_unknown, lib_X, lib_y)

    closed = float(np.mean(p_known == y_known))
    accept = s_known >= tau
    known_ok = float(np.mean((p_known == y_known) & accept))
    reject = float(np.mean(s_unknown < tau))

    n = len(y_known) + len(X_unknown)
    open_all = float(((p_known == y_known) & accept).sum() + (s_unknown < tau).sum()) / n
    auc = float(roc_auc_score(
        np.r_[np.ones(len(s_known)), np.zeros(len(s_unknown))],
        np.r_[s_known, s_unknown]))
    return {"closed": closed, "known_ok": known_ok, "reject": reject,
            "open_all": open_all, "auroc": auc, "tau": tau,
            "n_lib": len(lib_y), "s_known": s_known, "s_unknown": s_unknown}


def run_random(ds: Dataset, X, y, groups, per_class: int, seed: int = 0) -> dict:
    """随机协议：已知类 70/30 划分，库建在 70% 上，查询含已知 30% 与全部未知。"""
    rng = np.random.default_rng(seed)
    rows = []
    for unknown in np.unique(y):
        known_idx = np.where(y != unknown)[0]
        unk_idx = np.where(y == unknown)[0]
        tr, te = [], []
        for c in np.unique(y[known_idx]):
            idx = known_idx[y[known_idx] == c]
            rng.shuffle(idx)
            n_te = max(1, int(round(len(idx) * 0.3)))
            te.extend(idx[:n_te])
            tr.extend(idx[n_te:])
        r = evaluate_once(X[tr], y[tr], X[np.array(te)], y[np.array(te)],
                          X[unk_idx], per_class, seed)
        rows.append(r)
    return {k: float(np.mean([r[k] for r in rows]))
            for k in ("closed", "known_ok", "reject", "open_all", "auroc", "n_lib")}


def run_loo(ds: Dataset, X, y, groups, per_class: int) -> dict:
    """严格协议：留一个批次；库建在其余批次上，且库中不含未知类。"""
    rows = []
    for unknown in np.unique(y):
        for g in sorted({g for g in groups if g}):
            te = np.where(groups == g)[0]
            tr = np.where((groups != g) & (groups != "") & (y != unknown))[0]
            if tr.size < 50 or te.size < 10:
                continue
            known = te[y[te] != unknown]
            unk = te[y[te] == unknown]
            if known.size < 5 or unk.size < 3:
                continue
            rows.append(evaluate_once(X[tr], y[tr], X[known], y[known], X[unk],
                                      per_class, seed=hash(g) % 1000))
    return {k: float(np.mean([r[k] for r in rows]))
            for k in ("closed", "known_ok", "reject", "open_all", "auroc", "n_lib")}


def plot_results(rand: dict[int, dict], loo: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax = axes[0]
    sizes = sorted(rand)
    ax.plot(sizes, [rand[s]["closed"] for s in sizes], "o-", lw=1.8,
            color="#54A24B", label="闭集准确率")
    ax.plot(sizes, [rand[s]["reject"] for s in sizes], "s-", lw=1.8,
            color="#E45756", label="库外拒识率")
    ax.plot(sizes, [rand[s]["open_all"] for s in sizes], "^-", lw=1.8,
            color="#4C78A8", label="开放集准确率")
    ax.set_xlabel("每类库谱条数")
    ax.set_ylabel("比例")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(sizes)
    ax.set_title("库规模的影响（随机划分）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels = ["闭集", "库外拒识", "开放集"]
    keys = ["closed", "reject", "open_all"]
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, [rand[max(sizes)][k] for k in keys], w,
           label=f"随机划分（每类 {max(sizes)} 条）", color="#54A24B")
    ax.bar(x + w / 2, [loo[k] for k in keys], w,
           label="留一批次（每类 3 条）", color="#4C78A8")
    for xi, k in enumerate(keys):
        ax.text(xi - w / 2, rand[max(sizes)][k] + 0.015,
                f"{rand[max(sizes)][k]:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, loo[k] + 0.015, f"{loo[k]:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("比例")
    ax.set_title("两种协议下的库匹配表现", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="谱库匹配可行性实验")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="bacterial")
    parser.add_argument("--loo-per-class", type=int, default=3)
    parser.add_argument("--sim", choices=["raw", "deriv"], default="raw",
                        help="相似度所用的谱形：raw 原始谱 / deriv 一阶导数")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    names = (["bacterial", "oil"] if args.dataset == "both" else [args.dataset])
    all_rows = ["dataset,protocol,per_class,n_lib,closed,known_ok,reject,open_all,auroc"]

    results = {}
    for name in names:
        ds = (load_bacterial(DATA_DIR / "Bacterial_data") if name == "bacterial"
              else load_oil(DATA_DIR / "Oil_data"))
        X, y, groups = prepare_matrix(ds, normalize=None)
        X = row_snv(X)
        if args.sim == "deriv":
            X = np.diff(X, axis=1)   # 一阶导数去除残余背景，突出窄峰
        groups = np.asarray(groups)
        print("=" * 78)
        print(f"数据集：{name}（{len(ds)} 条，{len(ds.classes)} 类）")
        print(f"{'每类库谱':>8}{'闭集准确率':>12}{'库外拒识率':>12}{'开放集准确率':>14}{'AUROC':>9}")
        rand = {}
        for pc in LIB_SIZES:
            rand[pc] = run_random(ds, X, y, groups, pc)
            r = rand[pc]
            print(f"{pc:>8}{r['closed']:>12.3f}{r['reject']:>12.3f}"
                  f"{r['open_all']:>14.3f}{r['auroc']:>9.3f}")
            all_rows.append(f"{name},random,{pc},{r['n_lib']:.1f},"
                            f"{r['closed']:.4f},{r['known_ok']:.4f},{r['reject']:.4f},"
                            f"{r['open_all']:.4f},{r['auroc']:.4f}")
        loo = run_loo(ds, X, y, groups, args.loo_per_class)
        print(f"{'留批次':>8}{loo['closed']:>12.3f}{loo['reject']:>12.3f}"
              f"{loo['open_all']:>14.3f}{loo['auroc']:>9.3f}")
        all_rows.append(f"{name},loo_batch,{args.loo_per_class},{loo['n_lib']:.1f},"
                        f"{loo['closed']:.4f},{loo['known_ok']:.4f},{loo['reject']:.4f},"
                        f"{loo['open_all']:.4f},{loo['auroc']:.4f}")
        results[name] = (rand, loo)

    out_png = ROOT / f"library_match_result_{args.sim}.png"
    plot_results(results[names[0]][0], results[names[0]][1], out_png)
    (ROOT / f"library_match_result_{args.sim}.csv").write_text("\n".join(all_rows) + "\n",
                                                  encoding="utf-8-sig")
    print("表已保存:", ROOT / f"library_match_result_{args.sim}.csv")
    print("=" * 78)
    print("判读：留批次一列才是真实部署的数字；开放集准确率同时扣掉了误判与漏拒。")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
