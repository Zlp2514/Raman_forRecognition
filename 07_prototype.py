"""P5：原型分类能否替代 SVM 提升跨批次泛化。

核心问题
    SVM 学判别边界，边界由支持向量决定；跨批次时支持向量容易被偏移带偏。
    原型分类学类中心，没有边界概念，理论上对整体平移不敏感。
    如果原型法明显更好，说明瓶颈在决策边界，而不是特征表示。

用法
    .venv/Scripts/python.exe 07_prototype.py
    .venv/Scripts/python.exe 07_prototype.py --dataset oil
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
    ("原型·欧氏", "centroid"),
    ("原型·余弦", "centroid_cos"),
    ("原型·白化欧氏", "centroid_white"),
    ("kNN（k=5）", "knn"),
]


def build_classifier(kind: str):
    if kind == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="linear", C=1.0)
    if kind == "centroid":
        return NearestPrototype(metric="euclidean")
    if kind == "centroid_cos":
        return NearestPrototype(metric="cosine")
    if kind == "centroid_white":
        return NearestPrototype(metric="euclidean", whiten=True)
    if kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(kind)


def fit_predict(X_tr, y_tr, X_te, y_te, kind: str, n_components: int = 20) -> float:
    """统一流程：SNV → 标准化 → PCA → 分类器。所有配置共用同一套降维。"""
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    Zs_raw, Zt_raw = row_snv(X_tr), row_snv(X_te)
    scaler = StandardScaler().fit(Zs_raw)
    Zs, Zt = scaler.transform(Zs_raw), scaler.transform(Zt_raw)
    k = int(min(n_components, Zs.shape[0] - 1, Zs.shape[1]))
    pca = PCA(n_components=k, random_state=0).fit(Zs)
    Ps, Pt = pca.transform(Zs), pca.transform(Zt)

    clf = build_classifier(kind).fit(Ps, y_tr)
    return float(accuracy_score(y_te, clf.predict(Pt)))


def evaluate(dataset: Dataset, n_components: int = 20) -> list[dict]:
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
        acc_rand = fit_predict(X_raw[tr_idx], y[tr_idx], X_raw[te_idx], y[te_idx],
                               kind, n_components)
        fold_accs = [fit_predict(X_raw[tr], y[tr], X_raw[te], y[te], kind, n_components)
                     for _, tr, te in folds]
        rows.append({
            "config": name, "random": acc_rand,
            "loo_mean": float(np.mean(fold_accs)),
            "loo_std": float(np.std(fold_accs)),
            "folds": fold_accs,
        })
        print(f"  {name:14s} 随机 {acc_rand:.3f} | 留批次 "
              f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    return rows


def plot_rows(results: dict[str, list[dict]], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.6), squeeze=False)
    short = [c[0] for c in CONFIGS]
    x = np.arange(len(short))
    width = 0.38

    for ax, (ds_name, rows) in zip(axes[0], results.items()):
        rand = [r["random"] for r in rows]
        loo = [r["loo_mean"] for r in rows]
        ax.bar(x - width / 2, rand, width, label="随机划分", color="#54A24B")
        ax.bar(x + width / 2, loo, width, label="留一批次", color="#4C78A8",
               yerr=[r["loo_std"] for r in rows], capsize=3,
               error_kw={"lw": 1, "ecolor": "#333333"})
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=18, ha="right", fontsize=9)
        ax.set_ylim(0, 1.30)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("准确率")
        ax.set_title(f"{ds_name}：分类器对比", fontsize=11)
        ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="原型分类 vs SVM 的跨批次对比")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    results: dict[str, list[dict]] = {}
    if args.dataset in ("bacterial", "both"):
        print("=" * 72)
        print("数据集：bacterial（6 类菌，9 个批次）")
        results["bacterial"] = evaluate(load_bacterial(DATA_DIR / "Bacterial_data"),
                                        args.components)
    if args.dataset in ("oil", "both"):
        print("=" * 72)
        print("数据集：oil（8 类油，7 个批次）")
        results["oil"] = evaluate(load_oil(DATA_DIR / "Oil_data"), args.components)

    plot_rows(results, ROOT / "prototype_result.png")

    lines = ["dataset,config,random_acc,loo_mean,loo_std,gain_vs_svm"]
    for ds_name, rows in results.items():
        base = rows[0]["loo_mean"]
        for r in rows:
            lines.append(f"{ds_name},{r['config']},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f},"
                         f"{r['loo_mean'] - base:+.4f}")
    (ROOT / "prototype_result.csv").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8-sig")
    print("表已保存:", ROOT / "prototype_result.csv")
    print("=" * 72)
    print("判读：看 gain_vs_svm 在留批次协议下的值；明显为正说明瓶颈在决策边界。")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
