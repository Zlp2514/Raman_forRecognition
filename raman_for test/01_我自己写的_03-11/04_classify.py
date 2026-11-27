"""P2 验收脚本：在真实数据上跑分类，比较两种评估协议。

输出
    classification_result.png   左：随机划分的混淆矩阵；右：留一批次每折准确率
    classification_result.csv   两种协议的指标汇总

用法
    .venv/Scripts/python.exe 04_classify.py
    .venv/Scripts/python.exe 04_classify.py --dataset oil
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

from raman_rec import load_bacterial, load_oil, save_csv
from raman_rec.classify import (
    group_cv_eval,
    plot_confusion,
    random_split_eval,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def run(dataset, protocol_group: str, n_components: int, classifier: str) -> dict:
    """对单个数据集跑两种协议，返回结果字典。"""
    print("=" * 72)
    print(dataset.summary())
    print("-" * 72)

    rand = random_split_eval(dataset, test_size=0.3, seed=0,
                             n_components=n_components, classifier=classifier)
    print(f"[随机分层划分] 准确率 {rand['accuracy']:.3f} | "
          f"宏平均 F1 {rand['macro_f1']:.3f} | 测试样本 {rand['n']}")

    grouped = None
    if any(s.meta.get(protocol_group) for s in dataset.spectra):
        grouped = group_cv_eval(dataset, group_key=protocol_group,
                                n_components=n_components, classifier=classifier)
        print(f"[留一{protocol_group}验证] {len(grouped['folds'])} 折 | "
              f"平均准确率 {grouped['mean_accuracy']:.3f} "
              f"± {grouped['std_accuracy']:.3f}")
        for name, acc in zip(grouped["folds"], grouped["fold_accuracy"]):
            print(f"    留出 {name:>12s}  准确率 {acc:.3f}")
        gap = rand["accuracy"] - grouped["mean_accuracy"]
        print(f"[差距] 随机划分 - 留一组 = {gap:+.3f}"
              f"（差距越大，说明模型越依赖{protocol_group}带来的整体偏移）")

    return {"dataset": dataset, "random": rand, "grouped": grouped}


def plot_result(results: list[dict], out_path: Path) -> None:
    """每个数据集一行：左混淆矩阵，右各折准确率。"""
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5.2 * n), squeeze=False)

    for row, item in enumerate(results):
        ds = item["dataset"]
        rand = item["random"]
        plot_confusion(rand["confusion"], rand["labels"], axes[row][0],
                       title=f"{ds.name}：随机划分  acc={rand['accuracy']:.3f}")

        ax = axes[row][1]
        grouped = item["grouped"]
        if grouped:
            x = np.arange(len(grouped["folds"]))
            ax.bar(x, grouped["fold_accuracy"], color="#4C78A8", width=0.65)
            ax.axhline(grouped["mean_accuracy"], color="#E45756", ls="--", lw=1.5,
                       label=f"平均 {grouped['mean_accuracy']:.3f}")
            ax.axhline(rand["accuracy"], color="#54A24B", ls=":", lw=1.5,
                       label=f"随机划分 {rand['accuracy']:.3f}")
            ax.set_xticks(x)
            ax.set_xticklabels(grouped["folds"], rotation=45, ha="right", fontsize=8.5)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("准确率")
            ax.set_title(f"{ds.name}：留一批次验证", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25, axis="y")
        else:
            ax.text(0.5, 0.5, "该数据集无分组字段", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="拉曼光谱分类基线（PCA + 分类器）")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--components", type=int, default=20,
                        help="PCA 保留的主成分数")
    parser.add_argument("--classifier", choices=["svc", "lda", "logreg"],
                        default="svc")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    results = []
    if args.dataset in ("bacterial", "both"):
        results.append(run(load_bacterial(DATA_DIR / "Bacterial_data"),
                           "batch", args.components, args.classifier))
    if args.dataset in ("oil", "both"):
        # 油脂集没有批次字段，用官方 train/test 更公平，这里仍按随机划分报告
        results.append(run(load_oil(DATA_DIR / "Oil_data"),
                           "batch", args.components, args.classifier))

    plot_result(results, ROOT / "classification_result.png")

    lines = ["dataset,protocol,accuracy,macro_f1,n_samples"]
    for item in results:
        name = item["dataset"].name
        rand = item["random"]
        lines.append(f"{name},random_split,{rand['accuracy']:.6f},"
                     f"{rand['macro_f1']:.6f},{rand['n']}")
        if item["grouped"]:
            g = item["grouped"]
            lines.append(f"{name},leave_one_out,{g['mean_accuracy']:.6f},,{len(item['dataset'])}")
    (ROOT / "classification_result.csv").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8-sig")
    print("表已保存:", ROOT / "classification_result.csv")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
