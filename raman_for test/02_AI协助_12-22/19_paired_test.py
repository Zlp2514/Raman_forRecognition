"""P16：配对检验——那 0.6 个百分点的提升，到底算不算提升。

问题
    论文图 2 里，有三个配置比基线高 0.6–0.7 个百分点（波数平移、批次中心化、MLP），
    另有一个高 1.2 个百分点。这些差值在图上看起来"在噪声带边缘"，但噪声带只是视觉参考。

    正确的判据是**配对检验**：所有配置都在同样的 9 折上评估，
    因此可以对"每折的差值"做符号秩检验，而不是比较两个独立的均值。
    配对检验消除了"这一折本身难不难"这个共同变量，效力远高于独立比较。

做法
    1. 固定折划分（按批次留一），逐折计算每个配置的准确率；
    2. 对每个配置，求每折相对基线的差值；
    3. Wilcoxon 符号秩检验（双侧），报告 p 值与中位差值；
    4. 同时给出"该配置在几折上胜过基线"，便于判断方向是否一致。

用法
    .venv/Scripts/python.exe 19_paired_test.py
    .venv/Scripts/python.exe 19_paired_test.py --skip-slow    # 跳过 MLP 与增强
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
from raman_rec.transfer import batch_center, derivative, row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def folds_of(ds, groups):
    """按项目原有协议构建折（与 group_cv_eval 一致，避免自造划分）。"""
    from raman_rec.datasets import group_split

    id2idx = {id(sp): i for i, sp in enumerate(ds.spectra)}
    out = []
    for name, train, test in group_split(ds, group_key="batch"):
        tr = np.array([id2idx[id(sp)] for sp in train.spectra])
        te = np.array([id2idx[id(sp)] for sp in test.spectra])
        if tr.size and te.size:
            out.append((name, tr, te))
    return out


def _scale(X_tr, X_te):
    """与项目管线一致：先 StandardScaler 再分类。

    漏掉这一步会让基线从 0.8277 变成 0.8465（逐波数方差归一化本身就有判别作用）。
    """
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te)


def svm_acc(X_tr, y_tr, X_te, y_te) -> float:
    from sklearn.svm import SVC
    Xtr, Xte = _scale(X_tr, X_te)
    clf = SVC(kernel="linear", C=1.0).fit(Xtr, y_tr)
    return float(np.mean(clf.predict(Xte) == y_te))


def run_config(name: str, X: np.ndarray, y: np.ndarray, groups, folds,
               n_copies: int = 0) -> np.ndarray:
    """逐折计算某个配置的准确率。"""
    accs = []
    for g, tr, te in folds:
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

        if name == "收缩 LDA + SVM":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            from sklearn.svm import SVC
            Xtr, Xte = _scale(Xtr, Xte)
            lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
            # lsqr 求解器不提供 transform，用判别函数值作为降维后的特征
            Ftr, Fte = lda.decision_function(Xtr), lda.decision_function(Xte)
            clf = SVC(kernel="linear", C=1.0).fit(Ftr, ytr)
            acc = float(np.mean(clf.predict(Fte) == yte))

        elif name == "原型·欧氏":
            from raman_rec.prototype import NearestPrototype
            Xtr, Xte = _scale(Xtr, Xte)
            clf = NearestPrototype(metric="euclidean").fit(Xtr, ytr)
            acc = float(np.mean(clf.predict(Xte) == yte))

        elif name == "kNN（k=5）":
            from sklearn.neighbors import KNeighborsClassifier
            Xtr, Xte = _scale(Xtr, Xte)
            clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
            acc = float(np.mean(clf.predict(Xte) == yte))

        elif name == "MLP":
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xtr)
            clf = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3,
                                max_iter=200, early_stopping=True,
                                n_iter_no_change=15, random_state=0)
            clf.fit(sc.transform(Xtr), ytr)
            acc = float(np.mean(clf.predict(sc.transform(Xte)) == yte))

        elif name == "波数平移×4":
            from raman_rec.augment import augment
            Xa, ya, _ = augment(Xtr, ytr, np.array(["x"] * len(ytr)),
                                n_copies=n_copies, seed=0,
                                use_baseline=False, use_shift=True,
                                use_blur=False, use_noise=False)
            acc = svm_acc(Xa, ya, Xte, yte)   # svm_acc 内部已含标准化

        else:                      # 基线及仅改变特征的配置
            acc = svm_acc(Xtr, ytr, Xte, yte)
        accs.append(acc)
    return np.array(accs)


def main() -> int:
    parser = argparse.ArgumentParser(description="配对检验")
    parser.add_argument("--skip-slow", action="store_true",
                        help="跳过 MLP 与增强（耗时较长）")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    from scipy.stats import wilcoxon

    ds = load_bacterial(DATA_DIR / "Bacterial_data")
    X, y, groups = prepare_matrix(ds, normalize=None)
    X = row_snv(X)
    groups = np.asarray(groups)
    folds = folds_of(ds, groups)
    print(f"数据集：{len(ds)} 条，{len(ds.classes)} 类，{len(folds)} 折")

    x_axis = ds.spectra[0].x
    band_mask = (x_axis >= 500) & (x_axis <= 1800)

    variants: list[tuple[str, np.ndarray, int]] = [
        ("基线（SNV 全维）", X, 0),
        ("批次中心化", row_snv(batch_center(X, groups)), 0),
        ("指纹区筛选", X[:, band_mask], 0),
        ("一阶导数", derivative(X, order=1), 0),
        ("收缩 LDA + SVM", X, 0),
        ("原型·欧氏", X, 0),
        ("kNN（k=5）", X, 0),
    ]
    if not args.skip_slow:
        variants.append(("MLP", X, 0))
        variants.append(("波数平移×4", X, 4))

    per_fold: dict[str, np.ndarray] = {}
    for name, Xv, n_copies in variants:
        accs = run_config(name, Xv, y, groups, folds, n_copies=n_copies)
        per_fold[name] = accs
        print(f"  {name:16s} 均值 {accs.mean():.4f}  逐折 "
              + " ".join(f"{a:.3f}" for a in accs))

    base = per_fold["基线（SNV 全维）"]
    rows = []
    for name, accs in per_fold.items():
        if name == "基线（SNV 全维）":
            continue
        d = accs - base
        n_win = int((d > 0).sum())
        try:
            stat, p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        rows.append({"name": name, "mean": float(accs.mean()),
                     "delta": float(d.mean()), "median": float(np.median(d)),
                     "n_win": n_win, "p": float(p)})

    rows.sort(key=lambda r: -r["delta"])
    print("=" * 78)
    print(f"{'配置':16s}{'留批次':>9}{'Δ均值':>9}{'Δ中位':>9}{'胜出折数':>9}{'p 值':>10}{'显著':>6}")
    for r in rows:
        sig = "是" if r["p"] < 0.05 else "否"
        print(f"{r['name']:16s}{r['mean']:>9.4f}{r['delta']*100:>+9.2f}"
              f"{r['median']*100:>+9.2f}{r['n_win']:>6}/{len(folds)}"
              f"{r['p']:>10.4f}{sig:>6}")
    print("=" * 78)
    print("判读：p > 0.05 表示与基线的差异无法与折间波动区分；")
    print("      n=9 折时，Wilcoxon 双侧检验能达到的最小 p 约为 0.0039。")

    # ---------- 图：逐折差值的分布 ----------
    names = [r["name"] for r in rows]
    marks: list[str] = []
    fig, ax = plt.subplots(figsize=(9.4, max(4.2, 0.55 * len(names) + 1.6)))
    for i, r in enumerate(rows):
        d = (per_fold[r["name"]] - base) * 100
        jitter = np.linspace(-0.12, 0.12, len(d))
        color = "#4C78A8" if r["delta"] >= 0 else "#E45756"
        ax.scatter(d, np.full_like(d, i) + jitter, s=22, color=color, alpha=0.85)
        ax.plot([np.median(d)] * 2, [i - 0.28, i + 0.28], color="#333333", lw=2)
        marks.append(("n.s." if r["p"] >= 0.05 else f"p={r['p']:.3f}"))

    lo, hi = ax.get_xlim()
    pad = (hi - lo) * 0.22
    ax.set_xlim(lo, hi + pad)
    for i, mark in enumerate(marks):
        ax.text(hi + pad * 0.12, i, mark, fontsize=7.5, va="center",
                ha="left", color="#5F5E5A")
    ax.axvline(0, color="#333333", lw=1.1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("每折相对基线的准确率变化（百分点）　竖线为中位数")
    ax.set_title("配对比较：逐折差值的分布（n.s. 表示 p ≥ 0.05）", fontsize=10.5)
    ax.grid(alpha=0.22, axis="x")
    fig.tight_layout()
    out = ROOT / "paired_test.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("图已保存:", out)

    lines = ["config,loo_mean,delta_pt,median_delta_pt,n_folds_won,n_folds,p_value"]
    for r in rows:
        lines.append(f"{r['name']},{r['mean']:.4f},{r['delta']*100:+.2f},"
                     f"{r['median']*100:+.2f},{r['n_win']},{len(folds)},{r['p']:.4f}")
    (ROOT / "paired_test.csv").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8-sig")
    print("表已保存:", ROOT / "paired_test.csv")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
