"""统一运行入口：按脚本编号在 raman_for test/ 下定位并执行。

背景
    03–22 号实验脚本已迁到 `raman_for test/` 的两个子目录里，路径变长。
    这个入口让命令保持简短，并且自动把仓库根目录放进模块搜索路径。

用法
    python run.py 21 --no-show                  # 按编号
    python run.py 21_protocol_gap.py --no-show  # 按文件名
    python run.py 12 --stage apply --method asls
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
BASE = REPO / "raman_for test"


def locate(query: str) -> Path:
    """在归档目录里查找脚本；支持 "21" 与 "21_protocol_gap.py" 两种写法。"""
    if not BASE.exists():
        raise SystemExit(f"找不到归档目录：{BASE}")
    name = query if query.endswith(".py") else f"{query}"
    if not query.endswith(".py"):
        # 纯编号：匹配 NN_*.py
        hits = sorted(BASE.rglob(f"{query.zfill(2)}_*.py"))
    else:
        hits = sorted(BASE.rglob(name))
    if not hits:
        # 兜底：允许直接指定根目录下的脚本（01/02/compare 等）
        root_hits = sorted(REPO.glob(name))
        if root_hits:
            return root_hits[0]
        raise SystemExit(f"没有找到匹配 {query!r} 的脚本")
    if len(hits) > 1:
        listing = "\n  ".join(str(p.relative_to(REPO)) for p in hits)
        raise SystemExit(f"{query!r} 匹配到多个脚本，请指明：\n  {listing}")
    return hits[0]


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = locate(sys.argv[1])
    cmd = [sys.executable, str(script), *sys.argv[2:]]
    print(f"[run] {script.relative_to(REPO)}")
    return subprocess.call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    raise SystemExit(main())
