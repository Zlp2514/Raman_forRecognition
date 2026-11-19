"""pybaselines 入门复现：用模拟拉曼光谱比较 ModPoly 和 IModPoly。"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pybaselines
from pybaselines import Baseline, utils

OUTPUT_DIR = Path(__file__).resolve().parent
POLY_ORDER = 5

# 1. 构造横坐标、窄拉曼峰、宽荧光背景和噪声。
x = np.linspace(200, 2000, 1600)
true_signal = (
    utils.gaussian(x, 180, 520, 10)
    + utils.gaussian(x, 260, 1000, 16)
    + utils.gaussian(x, 150, 1450, 12)
    + utils.gaussian(x, 210, 1600, 20)
)
true_baseline = 250 + 900 * np.exp(-(x - 200) / 650) + 0.00035 * (x - 1000) ** 2
noise = np.random.default_rng(20261114).normal(0, 5, x.size)
y = true_signal + true_baseline + noise

# 2. 使用同一个 Baseline 对象分别拟合两种基线。
fitter = Baseline(x_data=x)
results = {}
for algorithm in ("modpoly", "imodpoly"):
    baseline, params = getattr(fitter, algorithm)(y, poly_order=POLY_ORDER)
    results[algorithm] = {
        "baseline": baseline,
        "corrected": y - baseline,
        "params": params,
    }

# 3. 保存原始模拟数据，供 02_process_my_csv.py 直接读取。
np.savetxt(
    OUTPUT_DIR / "demo_raman.csv",
    np.column_stack((x, y)),
    delimiter=",",
    header="raman_shift_cm-1,intensity",
    comments="",
)

# 4. 保存全部结果，便于用 Excel 或其他软件继续分析。
output_columns = [x, y, true_signal, true_baseline]
for algorithm in ("modpoly", "imodpoly"):
    output_columns.extend((results[algorithm]["baseline"], results[algorithm]["corrected"]))
np.savetxt(
    OUTPUT_DIR / "quickstart_corrected.csv",
    np.column_stack(output_columns),
    delimiter=",",
    header=("raman_shift_cm-1,raw_intensity,true_signal,true_baseline,"
            "modpoly_baseline,modpoly_corrected,imodpoly_baseline,imodpoly_corrected"),
    comments="",
)

# 5. 绘图并保存。IDLE 中按 F5 运行时还会弹出图形窗口。
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes[0].plot(x, y, color="0.55", lw=1, label="Synthetic measurement")
axes[0].plot(x, true_baseline, "k:", lw=2, label="True baseline")
for algorithm, color in (("modpoly", "tab:blue"), ("imodpoly", "tab:red")):
    axes[0].plot(x, results[algorithm]["baseline"], color=color,
                 lw=1.5, label=f"{algorithm} baseline")
axes[0].set_ylabel("Intensity")
axes[0].legend()

axes[1].plot(x, true_signal, "k:", lw=2, label="True Raman signal")
for algorithm, color in (("modpoly", "tab:blue"), ("imodpoly", "tab:red")):
    axes[1].plot(x, results[algorithm]["corrected"], color=color,
                 lw=1.2, label=f"{algorithm} corrected")
axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
axes[1].set_ylabel("Corrected intensity")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "quickstart_result.png", dpi=180)

print("Python:", sys.executable)
print("pybaselines:", pybaselines.__version__)
print("Polynomial order:", POLY_ORDER)
for algorithm in ("modpoly", "imodpoly"):
    rmse = np.sqrt(np.mean((results[algorithm]["baseline"] - true_baseline) ** 2))
    iterations = len(results[algorithm]["params"]["tol_history"])
    print(f"{algorithm}: baseline RMSE={rmse:.3f}, iterations={iterations}")
print("Results saved in:", OUTPUT_DIR)
plt.show()
