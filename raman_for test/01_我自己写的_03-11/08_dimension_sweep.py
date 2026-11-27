"""P5：维度扫描——PCA 到底帮了忙还是帮了倒忙。

起因
    P2–P4 的所有实验都把 PCA 固定为 20 维。原型分类实验后回头做敏感性检查，
    发现这个固定选择本身就是性能瓶颈：不做 PCA 时留批次反而更高。

    这是一个方法学教训：**当你把所有实验都固定在某一个超参上时，
    你验证的其实是"在这个超参下的结论"，而不是"结论本身"。**

本脚本做两件事
    A. 扫描 PCA 维数（10/20/50/100/200/不降维），两种协议都跑
    B. 在"不降维"条件下重跑 P3 的几种预处理，检验之前的负结论是否成立

用法
    .venv/Scripts/python.exe 08_dimension_sweep.py
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

from raman_rec import load_bacterial, load_oil, stratified_split
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.transfer import batch_center, derivative, row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DIMS = [10, 20, 50, 100, 200, None]          # None = 不降维


def fit_predict(X_tr, y_tr, X_te, y_te, n_components) -> float:
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    scaler = StandardScaler().fit(X_tr)
    Zs, Zt = scaler.transform(X_tr), scaler.transform(X_te)
    if n_components is not None:
        k = int(min(n_components, Zs.shape[0] - 1, Zs.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(Zs)
        Zs, Zt = pca.transform(Zs), pca.transform(Zt)
    clf = SVC(kernel="linear", C=1.0).fit(Zs, y_tr)
    return float(accuracy_score(y_te, clf.predict(Zt)))


def make_folds(dataset: Dataset):
    _, _, groups = prepare_matrix(dataset, normalize=None)
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
    return tr_idx, te_idx, folds, groups


def sweep_dimensions(dataset: Dataset) -> list[dict]:
    X_raw, y, _ = prepare_matrix(dataset, normalize=None)
    tr_idx, te_idx, folds, _ = make_folds(dataset)
    Z = row_snv(X_raw)

    rows = []
    for dim in DIMS:
        acc_rand = fit_predict(Z[tr_idx], y[tr_idx], Z[te_idx], y[te_idx], dim)
        accs = [fit_predict(Z[tr], y[tr], Z[te], y[te], dim) for _, tr, te in folds]
        rows.append({"dim": "不降维" if dim is None else str(dim),
                     "random": acc_rand,
                     "loo_mean": float(np.mean(accs)),
                     "loo_std": float(np.std(accs))})
        print(f"  PCA={str(dim):>5s}  随机 {acc_rand:.3f} | 留批次 "
              f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
    return rows


def recheck_preprocess(dataset: Dataset) -> list[dict]:
    """不降维条件下重跑 P3 的几种预处理。"""
    X_raw, y, groups = prepare_matrix(dataset, normalize=None)
    groups = np.asarray(groups)
    tr_idx, te_idx, folds, _ = make_folds(dataset)

    axis = dataset.spectra[0].x
    band = (axis >= 500.0) & (axis <= 1800.0)

    configs = [
        ("snv", lambda X, g: row_snv(X)),
        ("snv+band", lambda X, g: row_snv(X)[:, band]),
        ("snv+batch_center", lambda X, g: batch_center(row_snv(X), g)),
        ("snv+deriv", lambda X, g: derivative(row_snv(X))),
    ]

    rows = []
    for name, fn in configs:
        acc_rand = fit_predict(fn(X_raw[tr_idx], groups[tr_idx]), y[tr_idx],
                               fn(X_raw[te_idx], groups[te_idx]), y[te_idx], None)
        accs = [fit_predict(fn(X_raw[tr], groups[tr]), y[tr],
                            fn(X_raw[te], groups[te]), y[te], None)
                for _, tr, te in folds]
        rows.append({"config": name, "random": acc_rand,
                     "loo_mean": float(np.mean(accs)),
                     "loo_std": float(np.std(accs))})
        print(f"  {name:20s} 随机 {acc_rand:.3f} | 留批次 "
              f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
    return rows


def plot_sweep(results: dict[str, list[dict]], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.8), squeeze=False)
    for ax, (ds_name, rows) in zip(axes[0], results.items()):
        x = np.arange(len(rows))
        ax.plot(x, [r["random"] for r in rows], "o-", color="#54A24B",
                lw=1.8, label="随机划分")
        ax.plot(x, [r["loo_mean"] for r in rows], "s-", color="#4C78A8",
                lw=1.8, label="留一批次")
        ax.errorbar(x, [r["loo_mean"] for r in rows],
                    yerr=[r["loo_std"] for r in rows], fmt="none",
                    ecolor="#333333", elinewidth=1, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([r["dim"] for r in rows], fontsize=9)
        ax.set_xlabel("PCA 维数")
        ax.set_ylabel("准确率")
        ax.set_ylim(0.6, 1.02)
        ax.set_title(f"{ds_name}：降维对性能的影响", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA 维数扫描与预处理复检")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    sweep: dict[str, list[dict]] = {}
    recheck: dict[str, list[dict]] = {}

    if args.dataset in ("bacterial", "both"):
        print("=" * 72)
        print("[A] bacterial 维数扫描（6 类菌，9 批次）")
        ds = load_bacterial(DATA_DIR / "Bacterial_data")
        sweep["bacterial"] = sweep_dimensions(ds)
        print("-" * 72)
        print("[B] bacterial 不降维下重跑 P3 预处理")
        recheck["bacterial"] = recheck_preprocess(ds)

    if args.dataset in ("oil", "both"):
        print("=" * 72)
        print("[A] oil 维数扫描（8 类油，7 批次）")
        ds = load_oil(DATA_DIR / "Oil_data")
        sweep["oil"] = sweep_dimensions(ds)
        print("-" * 72)
        print("[B] oil 不降维下重跑 P3 预处理")
        recheck["oil"] = recheck_preprocess(ds)

    plot_sweep(sweep, ROOT / "dimension_sweep.png")

    lines = ["dataset,section,dim_or_config,random_acc,loo_mean,loo_std"]
    for ds_name, rows in sweep.items():
        for r in rows:
            lines.append(f"{ds_name},dim_sweep,{r['dim']},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f}")
    for ds_name, rows in recheck.items():
        for r in rows:
            lines.append(f"{ds_name},recheck_no_pca,{r['config']},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f}")
    (ROOT / "dimension_sweep.csv").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8-sig")
    print("表已保存:", ROOT / "dimension_sweep.csv")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
