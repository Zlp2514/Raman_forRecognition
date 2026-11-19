"""P11b：库匹配的改进版——峰位对齐 + 局部加权 + 分类别阈值。

上一轮的朴素实现（全谱相关 + 全局阈值）开放集准确率只有 0.45–0.56。
本轮针对三个已定位的缺陷各补一项：

    缺陷一 峰位随批次漂移，直接算相关等于比"谁漂得更像"
           → 对每条查询在 ±12 个采样点内平移，取最佳对齐后的相关系数

    缺陷二 全谱等权，判别信息被宽背景稀释
           → 按 Fisher 比（类间方差/类内方差）给每个波数加权，
             只用库内带标签样本估计，不涉及测试数据

    缺陷三 统一阈值，而不同物质可分性差别极大
           → 每个类别单独标定接受域阈值

三类指标同时报告：闭集准确率、库外拒识率、开放集准确率、AUROC。

用法
    .venv/Scripts/python.exe 14_library_match_v2.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil
from raman_rec.classify import prepare_matrix
from raman_rec.library import (best_shift_similarity, build_library,
                               class_thresholds, fisher_weights,
                               genuine_threshold, match)
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

MAX_SHIFT = 12          # ±12 个采样点，约 ±27 cm⁻¹
PER_CLASS = 3
KEEP_RATE = 0.95


def evaluate_split(X_lib, y_lib, X_known, y_known, X_unknown, variant: str) -> dict:
    """在一个划分上跑指定变体，返回四项指标。"""
    from sklearn.metrics import roc_auc_score

    if variant == "v1":
        lib_X, lib_y = build_library(X_lib, y_lib, per_class=PER_CLASS)
        tau = genuine_threshold(X_lib, y_lib, lib_X, lib_y, q=1 - KEEP_RATE)
        s_k, p_k = match(X_known, lib_X, lib_y)
        s_u, _ = match(X_unknown, lib_X, lib_y)
        acc_k = s_k >= tau
        acc_u = s_u >= tau
    else:
        w = np.sqrt(fisher_weights(X_lib, y_lib))
        lib_X, lib_y = build_library(X_lib * w, y_lib, per_class=PER_CLASS)
        taus = class_thresholds(X_lib * w, y_lib, lib_X, lib_y,
                                q=1 - KEEP_RATE, max_shift=6)

        def score(Q):
            """返回每条查询的最高相似度与对应类别（带峰位对齐）。"""
            sim = best_shift_similarity(Q * w, lib_X, MAX_SHIFT)
            idx = np.argmax(sim, axis=1)
            return sim[np.arange(len(idx)), idx], lib_y[idx]

        s_k, p_k = score(X_known)
        s_u, p_u = score(X_unknown)
        thr_k = np.array([taus.get(str(c), -np.inf) for c in p_k])
        thr_u = np.array([taus.get(str(c), -np.inf) for c in p_u])
        acc_k = s_k >= thr_k
        acc_u = s_u >= thr_u

    closed = float(np.mean(p_k == y_known))
    know_ok = float(np.mean((p_k == y_known) & acc_k))
    reject = float(np.mean(~acc_u))
    n = len(y_known) + len(s_u)
    open_all = float(((p_k == y_known) & acc_k).sum() + (~acc_u).sum()) / n
    auc = float(roc_auc_score(
        np.r_[np.ones(len(s_k)), np.zeros(len(s_u))], np.r_[s_k, s_u]))
    return {"closed": closed, "known_ok": know_ok, "reject": reject,
            "open_all": open_all, "auroc": auc}


def run_protocol(X, y, groups, protocol: str, variant: str) -> dict:
    """在给定协议下，对每个"未知类"跑一遍并取平均。"""
    rows = []
    if protocol == "random":
        rng = np.random.default_rng(0)
        for unknown in np.unique(y):
            known_idx = np.where(y != unknown)[0]
            unk_idx = np.where(y == unknown)[0]
            tr, te = [], []
            for c in np.unique(y[known_idx]):
                idx = known_idx[y[known_idx] == c]
                idx = idx.copy()
                rng.shuffle(idx)
                n_te = max(1, int(round(len(idx) * 0.3)))
                te.extend(idx[:n_te])
                tr.extend(idx[n_te:])
            te = np.array(te)
            rows.append(evaluate_split(X[tr], y[tr], X[te], y[te], X[unk_idx], variant))
    else:
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
                rows.append(evaluate_split(X[tr], y[tr], X[known], y[known], X[unk],
                                           variant))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def plot_compare(res: dict, out_path: Path) -> None:
    keys = ["closed", "reject", "open_all"]
    labels = ["闭集准确率", "库外拒识率", "开放集准确率"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for ax, proto, title in zip(axes, ("random", "loo_batch"),
                                ("随机划分", "留一批次")):
        x = np.arange(len(keys))
        w = 0.36
        v1 = [res[("v1", proto)][k] for k in keys]
        v2 = [res[("v2", proto)][k] for k in keys]
        ax.bar(x - w / 2, v1, w, label="朴素实现", color="#B0AFA8")
        ax.bar(x + w / 2, v2, w, label="改进版", color="#4C78A8")
        for xi, k in enumerate(keys):
            ax.text(xi - w / 2, v1[xi] + 0.015, f"{v1[xi]:.2f}", ha="center", fontsize=8.5)
            ax.text(xi + w / 2, v2[xi] + 0.015, f"{v2[xi]:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("比例")
        ax.set_title(f"{title}（每类 {PER_CLASS} 条库谱）", fontsize=11)
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="库匹配改进版对比")
    parser.add_argument("--dataset", choices=["bacterial", "oil"], default="bacterial")
    parser.add_argument("--keep-rate", type=float, default=0.95,
                        help="已知类接受率，越小阈值越松")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    global KEEP_RATE
    KEEP_RATE = args.keep_rate
    ds = (load_bacterial(DATA_DIR / "Bacterial_data") if args.dataset == "bacterial"
          else load_oil(DATA_DIR / "Oil_data"))
    X, y, groups = prepare_matrix(ds, normalize=None)
    X = row_snv(X)
    groups = np.asarray(groups)
    print(f"数据集：{args.dataset}（{len(ds)} 条，{len(ds.classes)} 类）")

    res = {}
    for variant in ("v1", "v2"):
        for proto in ("random", "loo_batch"):
            r = run_protocol(X, y, groups, proto, variant)
            res[(variant, proto)] = r
            print(f"  {variant} / {proto:9s} 闭集 {r['closed']:.3f} | "
                  f"拒识 {r['reject']:.3f} | 开放集 {r['open_all']:.3f} | "
                  f"AUROC {r['auroc']:.3f}")

    plot_compare(res, ROOT / f"library_v2_result_{args.dataset}.png")
    lines = ["variant,protocol,closed,known_ok,reject,open_all,auroc"]
    for (v, p), r in res.items():
        lines.append(f"{v},{p},{r['closed']:.4f},{r['known_ok']:.4f},"
                     f"{r['reject']:.4f},{r['open_all']:.4f},{r['auroc']:.4f}")
    out = ROOT / f"library_v2_result_{args.dataset}.csv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print("表已保存:", out)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
