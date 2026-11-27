"""P3：批次效应校正实验——能否把"留一批次"的准确率提上来。

比较六种预处理组合，在同一套划分下看两种协议的表现：

    raw              不处理（对照）
    snv              逐条标准化（P2 的基线做法）
    snv+deriv        加一阶导数，消加性偏移
    snv+batch_center 按批次减去该批平均谱
    snv+coral        PCA 后做 CORAL 域对齐
    snv+deriv+coral  导数 + CORAL

判读标准
    只看随机划分提升没有意义。真正的问题是：留一批次的那一列能不能从 0.778 往上走。
    如果某个方法在随机划分上没变、但留批次涨了，说明它确实在消除批次偏移，
    而不是在换一种方式拟合数据。

用法
    .venv/Scripts/python.exe 05_batch_effect.py
    .venv/Scripts/python.exe 05_batch_effect.py --dataset bacterial
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
from raman_rec.transfer import batch_center, coral_align, derivative, row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# (名称, 处理步骤序列, 是否在 PCA 之后做 CORAL 对齐)
#   snv           逐条标准正态变换
#   deriv         一阶 Savitzky-Golay 导数
#   band          只保留指纹区（500–1800 cm⁻¹）
#   batch_center  减去所属批次的平均谱
CONFIGS = [
    ("raw", [], False),
    ("snv", ["snv"], False),
    ("snv+deriv", ["snv", "deriv"], False),
    ("snv+batch_center", ["snv", "batch_center"], False),
    ("snv+coral", ["snv"], True),
    ("snv+deriv+coral", ["snv", "deriv"], True),
    ("snv+band", ["snv", "band"], False),
    ("snv+band+coral", ["snv", "band"], True),
]

BAND_RANGE = (500.0, 1800.0)


def apply_chain(X, groups, steps, band_mask, use_coral: bool):
    """按步骤序列处理特征矩阵；CORAL 需要目标域统计量，因此不在这里做。"""
    out = X
    for step in steps:
        if step == "snv":
            out = row_snv(out)
        elif step == "deriv":
            out = derivative(out)
        elif step == "band":
            out = out[:, band_mask]
        elif step == "batch_center":
            out = batch_center(out, groups)
        else:
            raise ValueError(f"未知步骤 {step!r}")
    return out, use_coral


def fit_predict(X_tr, y_tr, X_te, y_te, n_components: int = 20,
                use_coral: bool = False) -> float:
    """标准流程：标准化 → PCA →（可选 CORAL）→ 线性 SVM，返回测试准确率。"""
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    scaler = StandardScaler().fit(X_tr)
    Zs, Zt = scaler.transform(X_tr), scaler.transform(X_te)

    k = int(min(n_components, Zs.shape[0] - 1, Zs.shape[1]))
    pca = PCA(n_components=k, random_state=0).fit(Zs)
    Ps, Pt = pca.transform(Zs), pca.transform(Zt)

    if use_coral:
        Ps, Pt = coral_align(Ps, Pt)

    clf = SVC(kernel="linear", C=1.0).fit(Ps, y_tr)
    return float(accuracy_score(y_te, clf.predict(Pt)))


def evaluate_dataset(dataset: Dataset, group_key: str = "batch",
                     n_components: int = 20) -> list[dict]:
    """对每个预处理配置，跑随机划分与留一批次两种协议。"""
    X_raw, y, groups = prepare_matrix(dataset, normalize=None)
    groups = np.asarray(groups)

    # 固定一套随机划分的索引，保证各配置之间可比。
    # 注意：不能用 list.index()，Spectrum 是含 numpy 数组的 dataclass，
    # 相等比较会返回数组而报"truth value ambiguous"，必须按对象身份映射。
    train_ds, test_ds = stratified_split(dataset, test_size=0.3, seed=0)
    id2idx = {id(s): i for i, s in enumerate(dataset.spectra)}
    tr_idx = [id2idx[id(s)] for s in train_ds.spectra]
    te_idx = [id2idx[id(s)] for s in test_ds.spectra]

    # 留一批次的折索引
    folds = []
    for g in sorted({g for g in groups if g}):
        test_i = np.where(groups == g)[0]
        train_i = np.where((groups != g) & (groups != ""))[0]
        if test_i.size and train_i.size:
            folds.append((g, train_i, test_i))

    rows = []
    axis = dataset.spectra[0].x
    band_mask = (axis >= BAND_RANGE[0]) & (axis <= BAND_RANGE[1])

    for name, steps, use_coral in CONFIGS:
        # 随机划分
        X_tr, coral_tr = apply_chain(X_raw[tr_idx], groups[tr_idx], steps,
                                     band_mask, use_coral)
        X_te, _ = apply_chain(X_raw[te_idx], groups[te_idx], steps,
                              band_mask, use_coral)
        acc_rand = fit_predict(X_tr, y[tr_idx], X_te, y[te_idx],
                               n_components, coral_tr)

        # 留一批次
        fold_accs = []
        for _, train_i, test_i in folds:
            X_tr, coral_tr = apply_chain(X_raw[train_i], groups[train_i], steps,
                                         band_mask, use_coral)
            X_te, _ = apply_chain(X_raw[test_i], groups[test_i], steps,
                                  band_mask, use_coral)
            fold_accs.append(fit_predict(X_tr, y[train_i], X_te, y[test_i],
                                         n_components, coral_tr))

        rows.append({
            "config": name,
            "random": acc_rand,
            "loo_mean": float(np.mean(fold_accs)),
            "loo_std": float(np.std(fold_accs)),
            "folds": fold_accs,
        })
        print(f"  {name:20s} 随机 {acc_rand:.3f} | 留批次 "
              f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    return rows


def plot_rows(results: dict[str, list[dict]], out_path: Path) -> None:
    """每个数据集一个面板：横轴是预处理，两根柱是两种协议。"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 5.6), squeeze=False)
    names = [c[0] for c in CONFIGS]
    x = np.arange(len(names))
    width = 0.38

    for ax, (ds_name, rows) in zip(axes[0], results.items()):
        rand = [r["random"] for r in rows]
        loo = [r["loo_mean"] for r in rows]
        ax.bar(x - width / 2, rand, width, label="随机划分", color="#54A24B")
        ax.bar(x + width / 2, loo, width, label="留一批次", color="#4C78A8",
               yerr=[r["loo_std"] for r in rows], capsize=3,
               error_kw={"lw": 1, "ecolor": "#333333"})
        # 柱子上不再标数值：留批次有误差棒，数值标注会与之重叠；
        # 精确数字统一看 batch_effect_result.csv 与 README 的结果表。
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("准确率")
        ax.set_title(f"{ds_name}：不同预处理的两种协议", fontsize=11)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="批次效应校正实验")
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
        results["bacterial"] = evaluate_dataset(
            load_bacterial(DATA_DIR / "Bacterial_data"), n_components=args.components)

    if args.dataset in ("oil", "both"):
        print("=" * 72)
        print("数据集：oil（8 类油，7 个批次）")
        results["oil"] = evaluate_dataset(
            load_oil(DATA_DIR / "Oil_data"), n_components=args.components)

    plot_rows(results, ROOT / "batch_effect_result.png")

    lines = ["dataset,config,random_acc,loo_mean,loo_std,gain_vs_snv"]
    for ds_name, rows in results.items():
        base = next(r["loo_mean"] for r in rows if r["config"] == "snv")
        for r in rows:
            lines.append(f"{ds_name},{r['config']},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f},"
                         f"{r['loo_mean'] - base:+.4f}")
    (ROOT / "batch_effect_result.csv").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8-sig")
    print("表已保存:", ROOT / "batch_effect_result.csv")

    print("=" * 72)
    print("结论参考：以 snv 为基准，看『gain_vs_snv』一列在留批次协议下的变化。")
    print("若某配置随机划分基本不变、留批次明显上升，说明它真正消除了批次偏移。")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
