"""pybaselines 入门复现：用模拟拉曼光谱比较 ModPoly 和 IModPoly。

P0 重构说明
    数据生成与基线校正已移入 raman_rec 包：
        make_synthetic()    生成带真值的合成光谱
        remove_baseline()   七种基线方法的统一入口
    本脚本只保留"跑一遍、存结果、画一张图"的演示职责，
    因此依旧是理解整个流程的最好入口。
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pybaselines

from raman_rec import make_synthetic, remove_baseline, save_csv

OUTPUT_DIR = Path(__file__).resolve().parent
POLY_ORDER = 5

# 1. 生成合成光谱：4 个窄拉曼峰 + 指数/二次荧光背景 + 固定种子噪声。
#    真值（true_signal / true_baseline）留在 meta 里，只有合成数据才有。
spec = make_synthetic()
x, y = spec.x, spec.y
true_signal = spec.meta["true_signal"]
true_baseline = spec.meta["true_baseline"]

# 2. 同一份数据分别用两种多项式迭代算法扣背景。
results = {}
for algorithm in ("modpoly", "imodpoly"):
    corrected = remove_baseline(spec, algorithm, poly_order=POLY_ORDER)
    results[algorithm] = {
        "baseline": corrected.meta["baseline"],
        "corrected": corrected.y,
        "iterations": corrected.meta["baseline_iterations"],
    }

# 3. 保存原始模拟数据，供 02_process_my_csv.py 直接读取。
save_csv(
    OUTPUT_DIR / "demo_raman.csv",
    {"raman_shift_cm-1": x, "intensity": y},
)

# 4. 保存全部结果，便于用 Excel 或其他软件继续分析。
save_csv(
    OUTPUT_DIR / "quickstart_corrected.csv",
    {
        "raman_shift_cm-1": x,
        "raw_intensity": y,
        "true_signal": true_signal,
        "true_baseline": true_baseline,
        "modpoly_baseline": results["modpoly"]["baseline"],
        "modpoly_corrected": results["modpoly"]["corrected"],
        "imodpoly_baseline": results["imodpoly"]["baseline"],
        "imodpoly_corrected": results["imodpoly"]["corrected"],
    },
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
    diff = results[algorithm]["baseline"] - true_baseline
    rmse = float((diff ** 2).mean() ** 0.5)
    print(f"{algorithm}: baseline RMSE={rmse:.3f}, "
          f"iterations={results[algorithm]['iterations']}")
print("Results saved in:", OUTPUT_DIR)
plt.show()
