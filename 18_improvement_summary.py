"""论文图 2：六类改进尝试的汇总——为什么它们都没能突破基线。

这张图要回答一个问题：**过去两年这个领域提出的一堆改进手段，在严格协议下到底有没有用？**

做图前必须先解决可比性
    历史上的实验并不都在同一设置下完成：
      · P3（浅层校正）与 P5（分类器替换）是在 PCA=20 下做的；
      · P4（增强）、P6（度量学习）、P8（非线性）、P10（选参）是全维。
    把两种设置的数值画在一张图上，等于把"降维损失"混进"方法效果"里。

    处理方式：
      · 统一以**全维 SNV + 线性 SVM** 为基线（0.8277）；
      · P3 取 `dimension_sweep.csv` 中 recheck_no_pca 的复核值；
      · P5 当年只做了 PCA=20，本脚本**按同一协议重新算一遍全维**；
      · 其余直接取各自结果文件，并在 CSV 里注明来源。

    每个数值都带来源标签，读者可以逐条核对。

用法
    .venv/Scripts/python.exe 18_improvement_summary.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

NOISE_PT = 0.5      # 小于 0.5 个百分点视为落在波动内

CATEGORY_COLOR = {
    "浅层校正": "#4C78A8",
    "训练期增强": "#54A24B",
    "分类器替换": "#E45756",
    "线性度量": "#B279A2",
    "非线性模型": "#EECA3B",
    "无真值选参": "#72B7B2",
}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def loo_for_config(ds: Dataset, n_components=None, band=None) -> float:
    """按统一协议（留一批次）重算一个配置的准确率。"""
    import copy

    from raman_rec.classify import group_cv_eval

    target = copy.copy(ds)
    if band:
        target.spectra = [s.slice(*band) for s in ds.spectra]
    r = group_cv_eval(target, group_key="batch", n_components=n_components)
    return float(r["mean_accuracy"])


def recompute_classifiers(ds: Dataset) -> list[dict]:
    """P5 当年只在 PCA=20 下做过，这里在全维上补算。

    三个原型变体 + kNN(k=5)，协议与其它实验一致（留一批次，9 折）。
    """
    from sklearn.neighbors import KNeighborsClassifier

    from raman_rec.prototype import NearestPrototype

    X, y, groups = prepare_matrix(ds, normalize=None)
    X = row_snv(X)
    groups = np.asarray(groups)

    models = {
        "原型·欧氏": lambda: NearestPrototype(metric="euclidean"),
        "原型·余弦": lambda: NearestPrototype(metric="cosine"),
        "原型·白化欧氏": lambda: NearestPrototype(metric="euclidean", whiten=True),
        "kNN（k=5）": lambda: KNeighborsClassifier(n_neighbors=5),
    }
    out = []
    for name, factory in models.items():
        accs = []
        for g in sorted({g for g in groups if g}):
            te = np.where(groups == g)[0]
            tr = np.where((groups != g) & (groups != ""))[0]
            if te.size == 0 or tr.size == 0:
                continue
            clf = factory().fit(X[tr], y[tr])
            accs.append(float(np.mean(clf.predict(X[te]) == y[te])))
        out.append({"category": "分类器替换", "method": name,
                    "loo": float(np.mean(accs)),
                    "source": "本脚本重算（全维）"})
        print(f"    重算 {name:12s} 留批次 {np.mean(accs):.4f}")
    return out


def collect(ds: Dataset) -> tuple[float, list[dict]]:
    """汇集各轮结果，全部归一到全维留批次协议。"""
    # 基线
    rows = read_csv_rows(ROOT / "dimension_sweep.csv")
    base = next((float(r["loo_mean"]) for r in rows
                 if r["dataset"] == "bacterial" and r["section"] == "recheck_no_pca"
                 and r["dim_or_config"] == "snv"), 0.8277)
    entries: list[dict] = []

    # P3 浅层校正（取全维复核）
    name_map = {"snv+deriv": "一阶导数", "snv+band": "指纹区筛选",
                "snv+batch_center": "批次中心化"}
    for r in rows:
        if r["dataset"] == "bacterial" and r["section"] == "recheck_no_pca" \
                and r["dim_or_config"] in name_map:
            entries.append({"category": "浅层校正",
                            "method": name_map[r["dim_or_config"]],
                            "loo": float(r["loo_mean"]),
                            "source": "dimension_sweep.csv (recheck_no_pca)"})

    # P4 增强
    for r in read_csv_rows(ROOT / "augmentation_result.csv"):
        if r["dataset"] == "bacterial" and r["config"] != "snv（无增强）":
            label = (r["config"].replace("snv + ", "")
                     .replace("增强×2", "随机增强×2")
                     .replace("增强×4", "随机增强×4")
                     .replace("仅背景形状×4", "背景形状×4")
                     .replace("仅平移×4", "波数平移×4"))
            entries.append({"category": "训练期增强", "method": label,
                            "loo": float(r["loo_mean"]),
                            "source": "augmentation_result.csv"})

    # P5 分类器替换（全维重算）
    entries.extend(recompute_classifiers(ds))

    # P6 度量学习
    for r in read_csv_rows(ROOT / "metric_learning_result.csv"):
        if r["dataset"] != "bacterial" or r["config"] == "SVM（基线）":
            continue
        label = r["config"].replace("LDA(收缩) + SVM", "收缩 LDA + SVM") \
                           .replace("LDA + SVM", "LDA + SVM")
        entries.append({"category": "线性度量", "method": label,
                        "loo": float(r["loo_mean"]),
                        "source": "metric_learning_result.csv"})

    # P8 非线性
    for r in read_csv_rows(ROOT / "neural_result.csv"):
        if r["dataset"] != "bacterial":
            continue
        entries.append({"category": "非线性模型", "method": r["model"],
                        "loo": float(r["loo_mean"]),
                        "source": "neural_result.csv"})

    # P10 选参（默认 vs 代理，取代理）
    for r in read_csv_rows(ROOT / "param_selection_apply_imodpoly.csv"):
        if r.get("scheme") == "proxy_selected":
            entries.append({"category": "无真值选参", "method": "代理选参*",
                            "loo": float(r["loo_accuracy"]),
                            "note": "含显式基线校正，与其它配置管线不同",
                            "source": "param_selection_apply_imodpoly.csv"})
    return base, entries


def plot(base: float, entries: list[dict], out_path: Path) -> None:
    entries = sorted(entries, key=lambda e: e["loo"])
    labels = [e["method"] for e in entries]
    deltas = [(e["loo"] - base) * 100 for e in entries]
    colors = [CATEGORY_COLOR.get(e["category"], "#888780") for e in entries]

    fig, ax = plt.subplots(figsize=(9.6, max(4.6, 0.36 * len(entries) + 1.4)))
    y = np.arange(len(entries))
    hatches = ["//" if e.get("note") else "" for e in entries]
    bars = ax.barh(y, deltas, color=colors, height=0.68)
    for patch, h in zip(bars, hatches):
        if h:
            patch.set_hatch(h)
            patch.set_edgecolor("#333333")

    for yi, (e, d) in zip(y, zip(entries, deltas)):
        ax.text(d + (0.25 if d >= 0 else -0.25), yi, f"{e['loo']:.4f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=8)

    ax.axvline(0, color="#333333", lw=1.1)
    ax.axvspan(-NOISE_PT, NOISE_PT, color="#DDDDDD", alpha=0.55, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"相对全维 SNV + 线性 SVM 基线的变化（百分点）　基线 = {base:.4f}")
    ax.set_title("六类改进尝试在留一批次协议下的效果", fontsize=11)
    ax.set_xlim(min(deltas) - 1.6, max(deltas) + 1.8)
    ax.grid(alpha=0.22, axis="x")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in CATEGORY_COLOR.values()]
    ax.legend(handles, CATEGORY_COLOR.keys(), fontsize=8, ncol=3,
              loc="upper left", framealpha=0.95)
    fig.text(0.5, -0.035,
             "灰带 = ±0.5 pt 波动；* 该配置含显式基线校正，与其它配置管线不同"
             "（其增益主要来自校正步骤，选参本身相对默认参数为 −0.2 pt）",
             fontsize=7.4, ha="center", va="top", color="#5F5E5A")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="六类改进汇总图")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    ds = load_bacterial(DATA_DIR / "Bacterial_data")
    print(f"数据集：细菌，{len(ds)} 条，{len(ds.classes)} 类")
    base, entries = collect(ds)
    print(f"基线（全维 SNV + SVM，留批次）：{base:.4f}")
    print(f"共 {len(entries)} 个配置")

    plot(base, entries, ROOT / "improvement_summary.png")

    lines = ["category,method,loo_mean,delta_pt,source"]
    for e in sorted(entries, key=lambda x: x["loo"]):
        lines.append(f"{e['category']},{e['method']},{e['loo']:.4f},"
                     f"{(e['loo'] - base) * 100:+.2f},{e['source']}")
    (ROOT / "improvement_summary.csv").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8-sig")
    print("表已保存:", ROOT / "improvement_summary.csv")

    better = [e for e in entries if e["loo"] - base > NOISE_PT / 100]
    print(f"超过基线 {NOISE_PT} pt 的配置数：{len(better)}")
    for e in better:
        print(f"  {e['category']} / {e['method']}: {(e['loo']-base)*100:+.2f} pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
