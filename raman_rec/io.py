"""两列 CSV 的读写。

格式约定
    第一列拉曼位移（cm⁻¹），第二列强度，第一行可以是表头也可以直接是数据。

编码处理
    Windows 上用 Excel / WPS 另存的 CSV 常见 GBK 编码，而 Python 脚本和
    GitHub 网页按 UTF-8 读。本模块按 UTF-8 → GBK → latin-1 的顺序尝试解码，
    避免因为一个中文表头就报 UnicodeDecodeError。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectrum import Spectrum

_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin-1")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", raw, 0, 1, f"无法识别 {path} 的编码")


def _to_float(token: str) -> float | None:
    try:
        return float(token.strip())
    except (TypeError, ValueError):
        return None


def load_csv(path: str | Path) -> Spectrum:
    """读取两列 CSV，返回 Spectrum。

    容错规则
        - 表头行（前两列都不是数字）自动跳过
        - 空行忽略
        - 行内多余列忽略，只取前两列
        - 非法行跳过并计数，超过 10% 时报错，避免静默丢数据
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    text = _read_text(path)
    xs: list[float] = []
    ys: list[float] = []
    bad = 0
    total = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        parts = line.replace(";", ",").split(",")
        if len(parts) < 2:
            bad += 1
            continue
        vx, vy = _to_float(parts[0]), _to_float(parts[1])
        if vx is None or vy is None:
            bad += 1          # 通常是表头，不当错误
            continue
        xs.append(vx)
        ys.append(vy)

    if not xs:
        raise ValueError(f"{path.name} 里没有解析到数字，确认是两列 CSV")
    if bad > max(3, total * 0.1):
        raise ValueError(f"{path.name} 有 {bad}/{total} 行无法解析，检查分隔符是否为逗号")

    return Spectrum(x=np.asarray(xs), y=np.asarray(ys), meta={"source": str(path)})


def save_csv(
    path: str | Path,
    columns: dict[str, np.ndarray],
    encoding: str = "utf-8",
    fmt: str = "%.18e",
) -> Path:
    """按列写入 CSV。

    columns 是 有序字典 {列名: 数组}，列顺序即字典顺序。
    默认写 UTF-8，保证在 GitHub 网页上中文表头不乱码；
    若要用 Excel 直接双击打开，可传 encoding="utf-8-sig"（带 BOM）。
    fmt 默认 "%.18e"，与 numpy 默认一致，保证全精度可复现。
    """
    path = Path(path)
    names = list(columns)
    if not names:
        raise ValueError("至少需要一列数据")
    arrays = [np.asarray(columns[n], dtype=float) for n in names]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"各列长度不一致：{ {n: a.size for n, a in zip(names, arrays)} }")

    data = np.column_stack(arrays)
    np.savetxt(path, data, delimiter=",", header=",".join(names),
               comments="", fmt=fmt, encoding=encoding)
    return path
