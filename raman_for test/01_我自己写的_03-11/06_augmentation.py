"""P4：训练期数据增强能否改善跨批次泛化。

思路
    既然"改输入"的浅层校正全部失效（P3），就换一条路：不改推理时的输入，
    而是在训练时制造大量"看起来像别的批次"的样本，逼迫分类器忽略这些方向。

    关键前提：加性偏移与乘性缩放在 SNV 之后不可见（见 raman_rec/augment.py 文档），
    所以增强只加能改变形状的扰动：背景形状、波数平移、轻度模糊、噪声。

判读标准
    与 P3 相同——只看留一批次那一列。随机划分涨而留批次不涨，说明只是记住了更多样本。

用法
    .venv/Scripts/python.exe 06_augmentation.py
    .venv/Scripts/python.exe 06_augmentation.py --dataset bacterial
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
from raman_rec.augment import augment
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# (名称, 副本数, 是否加背景形状, 是否平移, 是否模糊, 是否加噪)
CONFIGS = [
    ("snv（无增强）", 1, False, False, False, False),
    ("snv + 增强×2", 2, True, True, True, True),
    ("snv + 增强×4", 4, True, True, True, True),
    ("snv + 仅背景形状×4", 4, True, False, False, False),
    ("snv + 仅平移×4", 4, False, True, False, False),
]


def fit_predict(X_tr, y_tr, X_te, y_te, n_components=None) -> float:
    """n_components=None 表示不降维（P5 维度扫描发现降维会压低细菌集性能）。"""
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


def evaluate(dataset: Dataset, n_components=None) -> list[dict]:
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
    for name, copies, use_base, use_shift, use_blur, use_noise in CONFIGS:
        # ---- 随机划分 ----
        Xtr_a, ytr_a, _ = augment(X_raw[tr_idx], y[tr_idx], groups[tr_idx],
                                  n_copies=copies, seed=1,
                                  use_baseline=use_base, use_shift=use_shift,
                                  use_blur=use_blur, use_noise=use_noise)
        acc_rand = fit_predict(row_snv(Xtr_a), ytr_a,
                               row_snv(X_raw[te_idx]), y[te_idx], n_components)

        # ---- 留一批次（只增强训练集，绝不增强测试集）----
        fold_accs = []
        for _, train_i, test_i in folds:
            Xtr_a, ytr_a, _ = augment(X_raw[train_i], y[train_i], groups[train_i],
                                      n_copies=copies, seed=1,
                                      use_baseline=use_base, use_shift=use_shift,
                                      use_blur=use_blur, use_noise=use_noise)
            fold_accs.append(fit_predict(row_snv(Xtr_a), ytr_a,
                                         row_snv(X_raw[test_i]), y[test_i],
                                         n_components))

        rows.append({
            "config": name, "random": acc_rand,
            "loo_mean": float(np.mean(fold_accs)),
            "loo_std": float(np.std(fold_accs)),
            "train_size": int(copies * len(tr_idx)),
        })
        print(f"  {name:22s} 随机 {acc_rand:.3f} | 留批次 "
              f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    return rows


def plot_rows(results: dict[str, list[dict]], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 5.6), squeeze=False)
    # 画图用短标签，完整名称见 CSV；长中文标签旋转后容易超出画布被裁掉
    short = ["无增强", "增强×2", "增强×4", "仅形状×4", "仅平移×4"]
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
        ax.set_xticklabels(short, fontsize=9.5)
        ax.set_ylim(0, 1.30)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("准确率")
        ax.set_title(f"{ds_name}：数据增强（SNV 后训练）", fontsize=11)
        ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练期数据增强实验")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--components", type=int, default=None,
                        help="PCA 维数，省略则不降维（P5 发现降维压低细菌集性能）")
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

    plot_rows(results, ROOT / "augmentation_result.png")

    lines = ["dataset,config,random_acc,loo_mean,loo_std,train_size,gain_vs_noaug"]
    for ds_name, rows in results.items():
        base = rows[0]["loo_mean"]
        for r in rows:
            lines.append(f"{ds_name},{r['config']},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f},"
                         f"{r['train_size']},{r['loo_mean'] - base:+.4f}")
    (ROOT / "augmentation_result.csv").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8-sig")
    print("表已保存:", ROOT / "augmentation_result.csv")
    print("=" * 72)
    print("判读：看 gain_vs_noaug 在留批次协议下的值；随机划分涨而留批次不涨=只记住了更多样本。")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
