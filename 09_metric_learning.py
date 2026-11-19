"""P6：度量学习——先把类内压紧，再看分类器是否好用。

思路来源
    P5 发现原型分类全面劣于 SVM，判断是"类内在特征空间里不够紧凑"。
    那么正确的顺序应该是：**先学一个让类内紧凑、类间分离的度量，再分类**。

本脚本用线性判别分析（LDA）作为最小成本的度量学习：
    它在有监督下最大化"类间散度 / 类内散度"，把数据投影到 C-1 维，
    投影后同类呈球状分布——这正是原型分类所需要的假设。

如果 LDA + 原型 能追平甚至超过 SVM，说明"类内不紧凑"这个判断成立，
下一步就该用更强的度量学习（三元组损失 / 对比学习）继续推进。

对照配置
    SVM（基线）   不降维、不学度量
    LDA + SVM     学度量 + 判别边界
    LDA + 原型    学度量 + 类中心（检验假设）
    LDA + kNN     学度量 + 局部投票

用法
    .venv/Scripts/python.exe 09_metric_learning.py
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
from raman_rec.prototype import NearestPrototype
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

CONFIGS = [
    ("SVM（基线）", "svm"),
    ("LDA + SVM", "lda_svm"),
    ("LDA(收缩) + SVM", "lda_shrunk_svm"),
    ("LDA + 原型", "lda_proto"),
    ("LDA + kNN", "lda_knn"),
]


def fit_predict(X_tr, y_tr, X_te, y_te, kind: str) -> float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    Zs, Zt = row_snv(X_tr), row_snv(X_te)
    scaler = StandardScaler().fit(Zs)
    Zs, Zt = scaler.transform(Zs), scaler.transform(Zt)

    if kind.startswith("lda"):
        n_classes = len(np.unique(y_tr))
        n_comp = max(1, min(n_classes - 1, Zs.shape[1], Zs.shape[0] - 1))
        if kind == "lda_shrunk_svm":
            # 1376 维下类内散度矩阵病态，用 Ledoit-Wolf 收缩修正估计误差
            lda = LinearDiscriminantAnalysis(n_components=n_comp, solver="eigen",
                                             shrinkage="auto")
        else:
            lda = LinearDiscriminantAnalysis(n_components=n_comp, solver="svd")
        lda.fit(Zs, y_tr)
        Ps, Pt = lda.transform(Zs), lda.transform(Zt)
        if kind in ("lda_svm", "lda_shrunk_svm"):
            clf = SVC(kernel="linear", C=1.0)
        elif kind == "lda_proto":
            clf = NearestPrototype(metric="euclidean")
        else:
            clf = KNeighborsClassifier(n_neighbors=5)
    else:
        Ps, Pt = Zs, Zt
        clf = SVC(kernel="linear", C=1.0)

    clf.fit(Ps, y_tr)
    return float(accuracy_score(y_te, clf.predict(Pt)))


def evaluate(dataset: Dataset) -> list[dict]:
    X_raw, y, groups = prepare_matrix(dataset, normalize=None)
    groups = np.asarray(groups)

    train_ds, test_ds = stratified_split(dataset, test_size=0.3, seed=0)
    id2idx = {id(s): i for i, s in enumerate(dataset.spectra)}
    tr_idx = np.array([id2idx[id(s)] for s in train_ds.spectra])
    te_idx = np.array([id2idx[id(s)] for s in test_ds.spectra])

    folds = []
    for g in sorted({g for g in groups if g}):
        test_i = np.where(groups == g)[0]
        train_i = np.where((groups != g) & (groups != ""))[0]
        if test_i.size and train_i.size:
            folds.append((g, train_i, test_i))

    rows = []
    for name, kind in CONFIGS:
        acc_rand = fit_predict(X_raw[tr_idx], y[tr_idx], X_raw[te_idx], y[te_idx], kind)
        accs = [fit_predict(X_raw[tr], y[tr], X_raw[te], y[te], kind)
                for _, tr, te in folds]
        rows.append({"config": name, "random": acc_rand,
                     "loo_mean": float(np.mean(accs)),
                     "loo_std": float(np.std(accs))})
        print(f"  {name:14s} 随机 {acc_rand:.3f} | 留批次 "
              f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
    return rows


def plot_rows(results: dict[str, list[dict]], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.4), squeeze=False)
    short = [c[0] for c in CONFIGS]
    x = np.arange(len(short))
    width = 0.38

    for ax, (ds_name, rows) in zip(axes[0], results.items()):
        ax.bar(x - width / 2, [r["random"] for r in rows], width,
               label="随机划分", color="#54A24B")
        ax.bar(x + width / 2, [r["loo_mean"] for r in rows], width,
               label="留一批次", color="#4C78A8",
               yerr=[r["loo_std"] for r in rows], capsize=3,
               error_kw={"lw": 1, "ecolor": "#333333"})
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=18, ha="right", fontsize=9)
        ax.set_ylim(0, 1.30)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("准确率")
        ax.set_title(f"{ds_name}：LDA 度量学习", fontsize=11)
        ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="LDA 度量学习实验")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    results: dict[str, list[dict]] = {}
    if args.dataset in ("bacterial", "both"):
        print("=" * 72)
        print("数据集：bacterial（6 类菌，9 个批次）")
        results["bacterial"] = evaluate(load_bacterial(DATA_DIR / "Bacterial_data"))
    if args.dataset in ("oil", "both"):
        print("=" * 72)
        print("数据集：oil（8 类油，7 个批次）")
        results["oil"] = evaluate(load_oil(DATA_DIR / "Oil_data"))

    plot_rows(results, ROOT / "metric_learning_result.png")

    lines = ["dataset,config,random_acc,loo_mean,loo_std,gain_vs_svm"]
    for ds_name, rows in results.items():
        base = rows[0]["loo_mean"]
        for r in rows:
            lines.append(f"{ds_name},{r['config']},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f},"
                         f"{r['loo_mean'] - base:+.4f}")
    (ROOT / "metric_learning_result.csv").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8-sig")
    print("表已保存:", ROOT / "metric_learning_result.csv")
    print("=" * 72)
    print("判读：LDA+原型 若能追平 SVM，说明『类内不紧凑』的判断成立。")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
