"""加载真实拉曼数据集，打印概览并画出各类平均光谱。

这是 P1 的验收脚本：跑通它说明数据能读、标签对得上、类别分布合理，
后面 P2 的分类模型就可以直接从这里取数据。

用法
    .venv/Scripts/python.exe 03_load_dataset.py
    .venv/Scripts/python.exe 03_load_dataset.py --dataset oil
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import (
    group_split,
    load_bacterial,
    load_oil,
    official_split,
    stratified_split,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def class_mean(spectra, cls: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回某一类的平均光谱与标准差。"""
    ys = np.vstack([s.y for s in spectra if s.label == cls])
    return ys.mean(axis=0), ys.std(axis=0), ys


def plot_overview(pairs, out_path: Path) -> None:
    """按数据集分面板，每类画一条平均光谱（带 ±1σ 带）。"""
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 5.2))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (name, dataset) in zip(axes, pairs):
        for cls in dataset.classes:
            mean, std, _ = class_mean(dataset.spectra, cls)
            x = dataset.spectra[0].x
            ax.plot(x, mean, lw=1.2, label=cls)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, lw=0)
        ax.set_title(f"{name}：{len(dataset)} 条 / {len(dataset.classes)} 类", fontsize=12)
        ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
        ax.set_ylabel("强度")
        ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="加载并检查真实拉曼数据集")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    pairs = []
    if args.dataset in ("bacterial", "both"):
        ds = load_bacterial(args.data_dir / "Bacterial_data")
        print("=" * 72)
        print(ds.summary())
        print("-" * 72)
        print("批次（batch）:", ds.meta.get("batches"))
        train, test = stratified_split(ds, test_size=0.3, seed=0)
        print(f"分层划分      : train={len(train)}  test={len(test)}")
        folds = group_split(ds, group_key="batch")
        print(f"留一批次折叠  : {len(folds)} 折，"
              f"首折测试集 {len(folds[0][2])} 条（批次 {folds[0][0]}）" if folds else "")
        pairs.append(("Bacterial（单细胞）", ds))

    if args.dataset in ("oil", "both"):
        ds = load_oil(args.data_dir / "Oil_data")
        print("=" * 72)
        print(ds.summary())
        print("-" * 72)
        train, test = official_split(ds)
        print(f"官方划分      : train={len(train)}  test={len(test)}")
        print(f"训练集类别    : {train.counts()}")
        print(f"测试集类别    : {test.counts()}")
        pairs.append(("Oil（食用油脂）", ds))

    plot_overview(pairs, ROOT / "dataset_overview.png")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
