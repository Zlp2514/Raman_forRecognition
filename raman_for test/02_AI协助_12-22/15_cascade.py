"""P12：级联方案——库匹配做门控，分类器做判定。

动机
    前两轮的结论很清楚：
      · 库匹配（相关系数）做识别不行：细菌闭集 0.73、油脂 0.94，拒识 AUROC 也只有 0.55–0.85；
      · 训练好的线性 SVM 识别很强（细菌 0.83、油脂 0.90），但它是闭集模型，遇到库外样本必错。

    两者恰好互补：库匹配擅长"这条谱像不像我库里的东西"，SVM 擅长"在库内诸类中该判哪一类"。
    于是把它们串起来：

        待测谱 → [门控] 与库比对，相似度低于阈值 → 判"库外"，流程结束
                        ↓ 通过
                 [判定] 线性 SVM 给出具体类别

    门控负责召回（宁松勿严，漏掉真未知物比多拒几条更糟），判定负责精度。

三个对照
    svm      纯分类器，不拒识——遇到未知物会硬判，这是闭集模型的典型失效
    gate     纯库匹配（P11b 的改进版），只回答"库里有没有"
    cascade  门控 + 判定

用法
    .venv/Scripts/python.exe 15_cascade.py
    .venv/Scripts/python.exe 15_cascade.py --dataset oil --keep-rate 0.95
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
from raman_rec.library import (best_shift_similarity, build_library,
                               class_thresholds, fisher_weights)
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

MAX_SHIFT = 12
PER_CLASS = 3
KEEP_RATE = 0.95


def evaluate_split(X_lib, y_lib, X_known, y_known, X_unknown, keep_rate: float) -> dict:
    """一次划分上同时评估三种方案。"""
    from sklearn.svm import SVC

    # ---------- 门控：Fisher 加权 + 峰位对齐的库匹配 ----------
    w = np.sqrt(fisher_weights(X_lib, y_lib))
    lib_X, lib_y = build_library(X_lib * w, y_lib, per_class=PER_CLASS)
    taus = class_thresholds(X_lib * w, y_lib, lib_X, lib_y,
                            q=1 - keep_rate, max_shift=6)

    def gate(Q):
        sim = best_shift_similarity(Q * w, lib_X, MAX_SHIFT)
        idx = np.argmax(sim, axis=1)
        s = sim[np.arange(len(idx)), idx]
        cls = lib_y[idx]
        thr = np.array([taus.get(str(c), -np.inf) for c in cls])
        return s, cls, s >= thr

    _, cls_k, pass_k = gate(X_known)
    _, _, pass_u = gate(X_unknown)

    # ---------- 判定：线性 SVM ----------
    svm = SVC(kernel="linear", C=1.0).fit(X_lib, y_lib)
    pred_k = svm.predict(X_known)

    # ---------- 三种方案 ----------
    # 1) 纯分类器：不认识"未知"这个概念，全部硬判
    n_all = len(y_known) + len(X_unknown)
    svm_closed = float(np.mean(pred_k == y_known))
    svm_open = float((pred_k == y_known).sum() / n_all)

    # 2) 纯门控：通过门控的样本用"最近库谱的类别"作答
    gate_closed = float(np.mean((cls_k == y_known) & pass_k))
    gate_open = float(((cls_k == y_known) & pass_k).sum() + (~pass_u).sum()) / n_all

    # 3) 级联：通过门控的样本再用 SVM 判
    casc_known_ok = float(np.mean((pred_k == y_known) & pass_k))
    casc_open = float(((pred_k == y_known) & pass_k).sum() + (~pass_u).sum()) / n_all

    return {
        "svm_closed": svm_closed, "svm_open": svm_open,
        "gate_known": gate_closed, "gate_open": gate_open,
        "casc_known": casc_known_ok, "casc_open": casc_open,
        "pass_known": float(np.mean(pass_k)),
        "reject_unknown": float(np.mean(~pass_u)),
    }


def run_protocol(X, y, groups, protocol: str, keep_rate: float) -> dict:
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
            te = np.array(te)
            rows.append(evaluate_split(X[tr], y[tr], X[te], y[te], X[unk_idx], keep_rate))
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
                                           keep_rate))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def plot_compare(res: dict[str, dict], out_path: Path) -> None:
    datasets = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    for ax, proto, title in zip(axes, ("random", "loo_batch"),
                                ("随机划分", "留一批次")):
        schemes = ["svm", "gate", "casc"]
        labels = ["纯分类器\n(不拒识)", "纯库匹配", "级联\n(库匹配+SVM)"]
        x = np.arange(len(schemes))
        width = 0.8 / max(len(datasets), 1)
        colors = ["#4C78A8", "#E45790", "#54A24B"]

        for di, ds_name in enumerate(datasets):
            vals = [res[ds_name][(proto, s)] for s in schemes]
            pos = x + (di - (len(datasets) - 1) / 2) * width
            ax.bar(pos, vals, width, label=ds_name, color=colors[di % len(colors)])
            for xi, v in zip(pos, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("开放集准确率")
        ax.set_title(f"{title}", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="库匹配 + 分类器级联")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"], default="both")
    parser.add_argument("--keep-rate", type=float, default=0.95)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    names = ["bacterial", "oil"] if args.dataset == "both" else [args.dataset]
    res: dict[str, dict] = {}
    lines = ["dataset,protocol,scheme,closed_open_accuracy,pass_known,reject_unknown"]

    for name in names:
        ds = (load_bacterial(DATA_DIR / "Bacterial_data") if name == "bacterial"
              else load_oil(DATA_DIR / "Oil_data"))
        X, y, groups = prepare_matrix(ds, normalize=None)
        X = row_snv(X)
        groups = np.asarray(groups)
        print("=" * 78)
        print(f"数据集：{name}（{len(ds)} 条，{len(ds.classes)} 类）")
        res[name] = {}
        for proto in ("random", "loo_batch"):
            r = run_protocol(X, y, groups, proto, args.keep_rate)
            res[name][(proto, "svm")] = r["svm_open"]
            res[name][(proto, "gate")] = r["gate_open"]
            res[name][(proto, "casc")] = r["casc_open"]
            print(f"  [{proto}] 纯分类器 {r['svm_open']:.3f}（闭集 {r['svm_closed']:.3f}）"
                  f" | 纯库匹配 {r['gate_open']:.3f}"
                  f" | 级联 {r['casc_open']:.3f}（闭集 {r['casc_known']:.3f}）"
                  f" | 门控放行 {r['pass_known']:.2f}/拒识 {r['reject_unknown']:.2f}")
            for key, val in (("svm", r["svm_open"]), ("gate", r["gate_open"]),
                             ("casc", r["casc_open"])):
                lines.append(f"{name},{proto},{key},{val:.4f},"
                             f"{r['pass_known']:.4f},{r['reject_unknown']:.4f}")

    plot_compare(res, ROOT / "cascade_result.png")
    (ROOT / "cascade_result.csv").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8-sig")
    print("表已保存:", ROOT / "cascade_result.csv")
    print("=" * 78)
    print("判读：开放集准确率 = 已知类判对且被放行 + 未知类被拒，按样本数加权。")
    print("      纯分类器不知道怎么拒识，遇到未知物一律硬判，所以它的开放集准确率最低。")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
