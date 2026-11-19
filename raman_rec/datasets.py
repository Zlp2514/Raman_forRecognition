"""真实拉曼光谱数据集加载器。

当前支持的数据集
    bacterial —— 单个细菌细胞，6 类菌种，2710 条，1376 个波数点
                 来源：Zenodo 15279321（Leibniz Institute of Photonic Technology, Guo Shuxia）
    oil       —— 食用油脂，8 类，560 条，916 个波数点，官方已划分 train/test
                 来源：同上

数据格式说明（两个数据集都是"矩阵式"CSV，不是一行一条光谱）
    第一行（或多个）是拉曼位移轴，之后每一行是一条光谱。
    这点与常见的"两列 CSV"完全不同，所以需要专门的解析器，
    不能用 raman_rec.io.load_csv。

许可证
    两个数据集均为 CC-BY-4.0，使用时必须注明来源：
    Guo, S. et al. Raman spectral datasets from single bacterial cells and different oil.
    Zenodo (2025). doi:10.5281/zenodo.15279321
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .spectrum import Spectrum

ZENODO_ID = 15279321
ZENODO_DOI = "10.5281/zenodo.15279321"
LICENSE = "CC-BY-4.0"
CITATION = ("Guo, S. Raman spectral datasets from single bacterial cells and different oil. "
            f"Zenodo (2025). doi:{ZENODO_DOI}")
DOWNLOAD_URL = f"https://zenodo.org/api/records/{ZENODO_ID}/files"


@dataclass
class Dataset:
    """一个光谱数据集：一组 Spectrum + 名称 + 附加信息。"""

    spectra: list[Spectrum]
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx]

    def __iter__(self):
        return iter(self.spectra)

    # ---------------------------------------------------------------- 类别信息
    @property
    def classes(self) -> list[str]:
        return sorted({s.label for s in self.spectra if s.label is not None})

    def labels(self) -> list[str]:
        return [s.label or "" for s in self.spectra]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.spectra:
            key = s.label or "未标注"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def is_balanced(self, tol: float = 0.05) -> bool:
        """各类样本数是否大致均衡（最大与最小之差不超过均值的 tol 倍）。"""
        counts = list(self.counts().values())
        if not counts:
            return True
        mean = float(np.mean(counts))
        return (max(counts) - min(counts)) <= tol * mean

    # ---------------------------------------------------------------- 子集操作
    def subset(self, indices) -> "Dataset":
        picked = [self.spectra[i] for i in indices]
        return Dataset(spectra=picked, name=self.name,
                       meta={**self.meta, "subset_of": self.name,
                             "subset_size": len(picked)})

    def filter(self, predicate) -> "Dataset":
        return self.subset([i for i, s in enumerate(self.spectra) if predicate(s)])

    # ---------------------------------------------------------------- 概览
    def summary(self) -> str:
        if not self.spectra:
            return f"{self.name}: 空数据集"
        n_points = {len(s) for s in self.spectra}
        lows = {round(float(s.x[0]), 1) for s in self.spectra}
        highs = {round(float(s.x[-1]), 1) for s in self.spectra}
        lines = [
            f"数据集      : {self.name}",
            f"样本数      : {len(self)}",
            f"类别数      : {len(self.classes)}  {', '.join(self.classes)}",
            f"每类样本    : {self.counts()}",
            f"点数        : {sorted(n_points) if len(n_points) <= 3 else '不一致'}",
            f"位移范围    : {sorted(lows)} ~ {sorted(highs)} cm-1",
            f"类别均衡    : {'是' if self.is_balanced() else '否'}",
        ]
        if "citation" in self.meta:
            lines.append(f"引用        : {self.meta['citation']}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 解析工具
def _read_matrix(path: Path, header_rows: int) -> np.ndarray:
    """读取纯数字矩阵，跳过前 header_rows 行表头。

    这两个数据集的 CSV 里，部分数值被双引号包裹（如 "1",400,402,...），
    因此必须显式指定 quotechar，否则 numpy 会把 '"1"' 当成非数字。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    data = np.loadtxt(path, delimiter=",", skiprows=header_rows,
                      dtype=float, quotechar='"')
    if data.ndim != 2:
        raise ValueError(f"{path.name} 不是二维矩阵，检查分隔符")
    return data


def load_bacterial(root: str | Path = "data/Bacterial_data") -> Dataset:
    """加载细菌单细胞数据集。

    文件布局
        data.csv      第 1 行是列名（V1…），第 2 行是拉曼位移轴，第 3 行起是光谱；
                      第 1 列是样本编号，其余列是强度
        metadata.csv  第 1 列是样本编号，另有 labels（菌种）与 batches（测量批次）

    batches 很重要：同一菌种分布在 9 个批次里，跨批次泛化是真实场景的难点，
    可以直接用 group_split(group_key="batch") 做留一批次交叉验证。
    """
    root = Path(root)
    matrix = _read_matrix(root / "data.csv", header_rows=1)

    axis = matrix[0, 1:]                 # 第 2 行，去掉首列的编号
    body = matrix[1:]                    # 第 3 行起
    sample_ids = body[:, 0].astype(int)
    intensities = body[:, 1:]

    import csv
    with open(root / "metadata.csv", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)               # 跳过表头 '"","labels","batches"'
        meta_rows = [row for row in reader if len(row) >= 3]

    # 关键：按行位置对齐，不能按编号查表。
    # data.csv 的第 1 行是指标轴，占用了编号 1，因此光谱编号是从 2 开始的
    # （实测 2..2711），而 metadata.csv 的编号是 1..2710，两边正好差 1。
    # 早期版本按编号 join，结果有 1 条光谱找不到标签。
    if len(meta_rows) != intensities.shape[0]:
        raise ValueError(
            f"光谱数 {intensities.shape[0]} 与元数据行数 {len(meta_rows)} 不一致")
    offset = int(sample_ids[0]) - int(float(meta_rows[0][0]))
    if offset != 1:
        raise ValueError(f"编号偏移异常（预期 1，实际 {offset}），请检查数据版本")

    spectra: list[Spectrum] = []
    for (sid, y, row) in zip(sample_ids, intensities, meta_rows):
        spectra.append(Spectrum(
            x=axis, y=y, label=row[1].strip() or None,
            meta={"sample_id": int(sid), "batch": row[2].strip(),
                  "dataset": "bacterial"},
        ))

    return Dataset(
        spectra=spectra, name="bacterial",
        meta={"n_points": axis.size, "citation": CITATION, "license": LICENSE,
              "source": f"Zenodo {ZENODO_ID}", "batches": sorted(
                  {s.meta.get("batch") for s in spectra if s.meta.get("batch")})},
    )


def load_oil(root: str | Path = "data/Oil_data") -> Dataset:
    """加载食用油脂数据集。

    文件布局
        spec.csv      第 1 行就是拉曼位移轴（无表头），第 2 行起是光谱
        metadata.csv  与光谱逐行对应：batches / labels / category(category=train|test)

    官方已给出 train/test 划分，直接沿用可比性最好；
    若要自己重新划分，用 stratified_split()。
    """
    root = Path(root)
    matrix = _read_matrix(root / "spec.csv", header_rows=0)

    axis = matrix[0]                     # 第 1 行就是轴
    intensities = matrix[1:]

    import csv
    with open(root / "metadata.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != intensities.shape[0]:
        raise ValueError(f"光谱数 {intensities.shape[0]} 与元数据行数 {len(rows)} 不一致")

    spectra: list[Spectrum] = []
    for y, rec in zip(intensities, rows):
        spectra.append(Spectrum(
            x=axis, y=y, label=rec.get("labels"),
            meta={"batch": rec.get("batches"), "split": rec.get("category"),
                  "dataset": "oil"},
        ))

    return Dataset(
        spectra=spectra, name="oil",
        meta={"n_points": axis.size, "citation": CITATION, "license": LICENSE,
              "source": f"Zenodo {ZENODO_ID}",
              "official_split": True},
    )


# ---------------------------------------------------------------- 划分
def stratified_split(
    dataset: Dataset,
    test_size: float = 0.3,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """按类别分层划分训练集与测试集，保证两边类别比例一致。"""
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 之间")
    rng = np.random.default_rng(seed)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in dataset.classes:
        idx = [i for i, s in enumerate(dataset.spectra) if s.label == cls]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train = Dataset(spectra=[dataset.spectra[i] for i in sorted(train_idx)],
                    name=f"{dataset.name}-train",
                    meta={**dataset.meta, "split": "train"})
    test = Dataset(spectra=[dataset.spectra[i] for i in sorted(test_idx)],
                   name=f"{dataset.name}-test",
                   meta={**dataset.meta, "split": "test"})
    return train, test


def official_split(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """使用数据集自带的 train/test 标记划分（仅 oil 有）。"""
    if not any("split" in s.meta for s in dataset.spectra):
        raise ValueError("该数据集没有自带划分，请用 stratified_split()")
    train = dataset.filter(lambda s: s.meta.get("split") == "train")
    test = dataset.filter(lambda s: s.meta.get("split") == "test")
    return (Dataset(train.spectra, f"{dataset.name}-train", {**dataset.meta, "split": "train"}),
            Dataset(test.spectra, f"{dataset.name}-test", {**dataset.meta, "split": "test"}))


def group_split(
    dataset: Dataset,
    group_key: str = "batch",
) -> list[tuple[str, Dataset, Dataset]]:
    """留一组交叉验证：每次留出一个批次做测试，其余批次做训练。

    光谱分类里真正难的往往不是"分得开不同物质"，而是"换了测量批次还分得开"。
    用这个划分才能看出模型有没有学到真实光谱特征、而不是记住了批次偏移。
    """
    groups = sorted({s.meta.get(group_key) for s in dataset.spectra
                     if s.meta.get(group_key)})
    folds = []
    for g in groups:
        train = dataset.filter(lambda s: s.meta.get(group_key) != g)
        test = dataset.filter(lambda s: s.meta.get(group_key) == g)
        if test.spectra:
            folds.append((g, train, test))
    return folds
