"""读取两列 CSV，使用 IModPoly 消减荧光背景并保存结果。

P0 重构说明
    CSV 读取改用 raman_rec.load_csv()，它会自动处理表头与 GBK/UTF-8 编码差异；
    基线校正走 raman_rec.remove_baseline()，与 01_quickstart.py 使用同一份实现。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from raman_rec import load_csv, remove_baseline, save_csv

# 初学者主要修改下面三个参数。
POLY_ORDER = 5
NUM_STD = 1.0
INPUT_NAME = "my_raman.csv"

BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / INPUT_NAME
if not input_path.exists():
    # 没有自己的数据时，自动使用 01_quickstart.py 生成的演示数据。
    input_path = BASE_DIR / "demo_raman.csv"
if not input_path.exists():
    raise FileNotFoundError("请先运行 01_quickstart.py，或提供 my_raman.csv。")

# load_csv 会自动跳过表头、按拉曼位移排序、检查重复值与 NaN。
spec = load_csv(input_path)
x, y = spec.x, spec.y

corrected = remove_baseline(spec, "imodpoly", poly_order=POLY_ORDER, num_std=NUM_STD)
baseline = corrected.meta["baseline"]
corrected_y = corrected.y

output_path = input_path.with_name(input_path.stem + "_corrected.csv")
figure_path = input_path.with_name(input_path.stem + "_corrected.png")
save_csv(
    output_path,
    {
        "raman_shift_cm-1": x,
        "raw_intensity": y,
        "baseline": baseline,
        "corrected_intensity": corrected_y,
    },
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color="0.55", lw=1, label="Raw spectrum")
ax.plot(x, baseline, "tab:red", lw=1.8, label="IModPoly baseline")
ax.plot(x, corrected_y, "tab:blue", lw=1.1, label="Corrected spectrum")
ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend()
fig.tight_layout()
fig.savefig(figure_path, dpi=180)

print("Input:", input_path)
print("Points:", x.size, f"({x[0]:.1f}~{x[-1]:.1f} cm-1)")
print("Polynomial order:", POLY_ORDER, "; num_std:", NUM_STD)
print("Iterations:", corrected.meta["baseline_iterations"])
print("Negative ratio:", f"{np.mean(corrected_y < 0):.4f}")
print("CSV output:", output_path)
print("Figure output:", figure_path)
plt.show()
