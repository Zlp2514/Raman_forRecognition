"""P17：论文图 3——拒识能力的类别依赖性。

要传达的一句话
    "平均 AUROC" 这个数字没有决策意义：同一个模型、同一套评分器，
    换一个未知类，AUROC 可以从 0.10 跳到 0.96。

为什么必须重算
    P7 的报告里给过分类别数值，但没有落盘成结果文件。论文图必须可复现，
    因此本脚本按原协议重新计算，并把每一项存进 CSV。

设计
    留一类法：每次把 1 类排除在训练外当作未知物；已知类的测试样本与未知样本混在一起，
    计算三类评分器区分"已知 / 未知"的 AUROC。
    两种协议各跑一遍：随机划分（乐观）、留一批次（严格）。

用法
    .venv/Scripts/python.exe 20_openset_perclass.py
    .venv/Scripts/python.exe 20_openset_perclass.py --dataset oil
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

from raman_rec import load_bacterial, load_oil
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.openset import OpenSetModel
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCORERS = ["margin", "prototype", "mahalanobis"]
SCORER_CN = {"margin": "SVM 决策值", "prototype": "原型距离", "mahalanobis": "马氏距离"}
SCORER_MARKER = {"margin": "o", "prototype": "s", "mahalanobis": "^"}
SCORER_COLOR = {"margin": "#4C78A8", "prototype": "#54A24B", "mahalanobis": "#E45756"}


def auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    return float(roc_auc_score(y, np.r_[known_scores, unknown_scores]))


def evaluate(X_tr, y_tr, X_known, X_unknown) -> dict[str, float]:
    model = OpenSetModel().fit(X_tr, y_tr)
    s_k, s_u = model.score(X_known), model.score(X_unknown)
    return {k: auroc(s_k[k], s_u[k]) for k in SCORERS}


def run(ds: Dataset, protocol: str) -> list[dict]:
    X, y, groups = prepare_matrix(ds, normalize=None)
    X = row_snv(X)
    groups = np.asarray(groups)
    rows = []

    if protocol == "random":
        rng = np.random.default_rng(0)
        for unknown in np.unique(y):
            known_idx = np.where(y != unknown)[0]
            unk_idx = np.where(y == unknown)[0]
            tr, te = [], []
            for c in np.unique(y[known_idx]):
                idx = known_idx[y[known_idx] == c].copy()
                rng.shuffle(idx)
                n_te = max(1, int(round(len(idx) * 0.3)))
                te.extend(idx[:n_te])
                tr.extend(idx[n_te:])
            acc = evaluate(X[tr], y[tr], X[np.array(te)], X[unk_idx])
            rows.append({"unknown": str(unknown), "protocol": protocol, **acc})
    else:
        per_class: dict[str, list[dict]] = {}
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
                per_class.setdefault(str(unknown), []).append(
                    evaluate(X[tr], y[tr], X[known], X[unk]))
        for cls, lst in per_class.items():
            rows.append({"unknown": cls, "protocol": protocol,
                         **{k: float(np.mean([r[k] for r in lst])) for k in SCORERS}})
    return rows


def plot(results: dict[str, list[dict]], out_path: Path) -> None:
    datasets = list(results)
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5.6 * len(datasets) + 1.2, 5.4),
                             sharex=True, squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, datasets):
        rows = results[name]
        # 以马氏距离在两个协议下的均值排序
        order = {}
        for cls in {r["unknown"] for r in rows}:
            vals = [r["mahalanobis"] for r in rows if r["unknown"] == cls]
            order[cls] = float(np.mean(vals))
        classes = sorted(order, key=order.get)
        ypos = {c: i for i, c in enumerate(classes)}

        for r in rows:
            y = ypos[r["unknown"]]
            filled = r["protocol"] == "random"
            for j, k in enumerate(SCORERS):
                ax.scatter(r[k], y + (j - 1) * 0.24,
                           marker=SCORER_MARKER[k], s=46,
                           facecolor=SCORER_COLOR[k] if filled else "none",
                           edgecolor=SCORER_COLOR[k], linewidth=1.2, alpha=0.95)

        ax.axvline(0.5, color="#666666", ls="--", lw=1.1)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("AUROC（区分已知 / 未知）")
        ax.set_title(f"{name}（{len(classes)} 类）", fontsize=11)
        ax.grid(alpha=0.22, axis="x")

    # 图例：颜色=评分器，实心=随机划分，空心=留一批次
    handles = []
    for k in SCORERS:
        handles.append(plt.Line2D([], [], marker=SCORER_MARKER[k], ls="",
                                  markerfacecolor=SCORER_COLOR[k],
                                  markeredgecolor=SCORER_COLOR[k], markersize=7,
                                  label=SCORER_CN[k]))
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="#888780",
                              markeredgecolor="#888780", markersize=7,
                              label="实心：随机划分"))
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                              markeredgecolor="#888780", markersize=7,
                              label="空心：留一批次"))
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    fig.suptitle("拒识的类别依赖性：AUROC 随未知类从 0.1 到 0.97", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="分类别拒识能力")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"], default="both")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    names = ["bacterial", "oil"] if args.dataset == "both" else [args.dataset]
    results: dict[str, list[dict]] = {}
    lines = ["dataset,protocol,unknown_class,scorer,auroc"]
    for name in names:
        ds = (load_bacterial(DATA_DIR / "Bacterial_data") if name == "bacterial"
              else load_oil(DATA_DIR / "Oil_data"))
        print(f"数据集：{name}（{len(ds)} 条，{len(ds.classes)} 类）")
        rows = []
        for proto in ("random", "loo_batch"):
            rows.extend(run(ds, proto))
        results[name] = rows
        for r in sorted(rows, key=lambda x: (x["unknown"], x["protocol"])):
            for k in SCORERS:
                lines.append(f"{name},{r['protocol']},{r['unknown']},{k},{r[k]:.4f}")
        m = [r["mahalanobis"] for r in rows if r["protocol"] == "loo_batch"]
        print(f"  马氏距离·留批次：均值 {np.mean(m):.3f}，"
              f"范围 {np.min(m):.3f}–{np.max(m):.3f}")
        print("  分类别（留批次）：" + "、".join(
            f"{r['unknown'][:10]} {r['mahalanobis']:.2f}"
            for r in sorted(rows, key=lambda x: x["mahalanobis"])
            if r["protocol"] == "loo_batch"))

    plot(results, ROOT / "openset_perclass.png")
    (ROOT / "openset_perclass.csv").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8-sig")
    print("表已保存:", ROOT / "openset_perclass.csv")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
