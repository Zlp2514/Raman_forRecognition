"""读取两列 CSV，使用 IModPoly 消减荧光背景并保存结果。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pybaselines import Baseline

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

# CSV 第一列是拉曼位移，第二列是强度，第一行为表头。
data = np.loadtxt(input_path, delimiter=",", skiprows=1)
if data.ndim != 2 or data.shape[1] < 2:
    raise ValueError("输入文件必须至少包含两列：拉曼位移和强度。")
x, y = data[:, 0], data[:, 1]
if not np.all(np.isfinite(data[:, :2])):
    raise ValueError("输入数据含空值、NaN 或无穷值，请先清理。")

fitter = Baseline(x_data=x)
baseline, params = fitter.imodpoly(
    y, poly_order=POLY_ORDER, num_std=NUM_STD
)
corrected = y - baseline

output_path = input_path.with_name(input_path.stem + "_corrected.csv")
figure_path = input_path.with_name(input_path.stem + "_corrected.png")
np.savetxt(
    output_path,
    np.column_stack((x, y, baseline, corrected)),
    delimiter=",",
    header="raman_shift_cm-1,raw_intensity,baseline,corrected_intensity",
    comments="",
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color="0.55", lw=1, label="Raw spectrum")
ax.plot(x, baseline, "tab:red", lw=1.8, label="IModPoly baseline")
ax.plot(x, corrected, "tab:blue", lw=1.1, label="Corrected spectrum")
ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend()
fig.tight_layout()
fig.savefig(figure_path, dpi=180)

print("Input:", input_path)
print("Polynomial order:", POLY_ORDER, "; num_std:", NUM_STD)
print("Iterations:", len(params["tol_history"]))
print("CSV output:", output_path)
print("Figure output:", figure_path)
plt.show()
