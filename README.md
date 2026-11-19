# Raman_forRecognition

拉曼光谱荧光背景扣除（基线校正）的复现与横向对比，基于 [pybaselines](https://github.com/derb12/pybaselines)。

拉曼光谱测量中，样品本身的荧光会产生一条缓慢变化的背景，把窄的拉曼峰抬高、甚至淹没。这个项目用 pybaselines 提供的多种算法把这条背景估计出来并减掉，同时在同一份数据上横向比较六种方法的差异。

![六种基线方法对比](compare_result.png)

## 这个项目做了什么

| 脚本 | 作用 |
|---|---|
| `01_quickstart.py` | 用合成光谱（4 个高斯峰 + 指数/二次背景 + 噪声）复现 ModPoly 与 IModPoly，并计算基线误差 |
| `02_process_my_csv.py` | 读取自己准备的两列 CSV，用 IModPoly 做校正，输出结果谱与图 |
| `compare_baseline_methods.py` | 在同一个光谱上同时运行 **6 种方法**，对比基线形状、校正谱与误差 |

## 快速开始

```bash
# 1. 准备环境
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. 运行模拟数据复现
python 01_quickstart.py

# 3. 处理自己的数据（需先放 my_raman.csv 在同目录）
python 02_process_my_csv.py

# 4. 六种方法横向对比
python compare_baseline_methods.py
python compare_baseline_methods.py --csv my_raman.csv
```

## 输入数据格式

两列 CSV，第一列拉曼位移（cm⁻¹），第二列强度，**第一行为表头**：

```csv
raman_shift_cm-1,intensity
200,1250
201,1252
202,1260
```

## 六种方法的对比结果

合成光谱（1600 点，噪声标准差 5.0，真实背景已知）：

| 方法 | 最优参数 | 基线 RMSE |
|---|---|---|
| `imodpoly` | poly_order=5 | **0.727** |
| `mixture_model` | p=0.7 | 0.945 |
| `airpls` | lam=1e6 | 6.217 |
| `pspline_asls` | lam=1000, p=0.05 | 6.832 |
| `asls` | lam=1e6, p=0.05 | 7.000 |
| `snip` | max_half_window=40 | 9.301 |

> 共选取 6 个有代表性的方法。pybaselines 一维方法实际有 60 余个。

## 三个必须注意的结论

1. **RMSE 排名不等于方法优劣。** 这条合成谱的真实背景接近低阶多项式，所以 `imodpoly` 占优有数据配合的成分。真实样品的荧光形状往往不是多项式，`asls`、`airpls` 可能更稳。

2. **参数影响大于方法选择。** 同一个 `imodpoly`，`poly_order=2` 时 RMSE 约 13.8，`poly_order=5` 时是 0.727。换算法之前先调参。

3. **真实数据没有真值，不能用 RMSE 自动选参。** 合成数据的"自动挑最优参数"是它独有的特权。真实数据只能靠峰形是否被削、负值比例、重复谱一致性来判断。

## 目录结构

```
Raman_forRecognition/
├── raman_rec/                    # 核心代码包（P0 重构后新增）
│   ├── spectrum.py               #   Spectrum 数据结构：归一化/平滑/寻峰/信噪比
│   ├── preprocess.py             #   七种基线方法的统一入口 + 参数网格
│   ├── io.py                     #   两列 CSV 读写（自动处理 GBK/UTF-8 与表头）
│   └── synthetic.py              #   带真值的合成光谱生成器
├── 01_quickstart.py              # 演示：模拟数据 ModPoly vs IModPoly
├── 02_process_my_csv.py          # 实用：处理自备 CSV
├── compare_baseline_methods.py   # 对比：六种方法横向比较（支持 --csv）
├── experiments/README.md         # 实验记录规范与表格
├── my_raman.csv                  # 示例输入数据
├── quickstart_result.png         # 01 的输出图
├── my_raman_corrected.png        # 02 的输出图
├── compare_result.png            # 对比脚本的三联图
├── 操作说明.txt                   # 中文操作手册
├── requirements.txt
└── .gitignore
```

其余 `*_corrected.csv`、`demo_raman.csv` 等中间产物可由脚本重新生成，已在 `.gitignore` 中排除。

## 核心 API

所有处理都以 `Spectrum` 为中心，每一步返回新对象，可以链式调用：

```python
from raman_rec import load_csv, remove_baseline

spec = load_csv("my_raman.csv")                 # 读数据，自动跳过表头、处理 GBK
spec = remove_baseline(spec, "imodpoly", poly_order=5)
spec = spec.smoothed(window=7).normalized("snv")  # 平滑 + 标准化

peaks = spec.peaks(prominence_frac=0.05)        # 寻峰
print(peaks["shift"][:5])                       # 前五个峰位
print("SNR:", spec.snr())
print("负值比例:", spec.negative_ratio())
```

可用基线方法：`modpoly` / `imodpoly` / `asls` / `airpls` / `snip` / `mixture_model` / `pspline_asls`。

## 开发路线

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 代码重构为包结构，统一数据与预处理入口 | ✅ 已完成 |
| P1 | 接入真实光谱数据集（RRUFF），统一格式与划分 | ⬜ 待开始 |
| P2 | PCA + SVM 分类基线，跑通第一个识别准确率 | ⬜ |
| P3 | 1D-CNN 对照实验 | ⬜ |
| P4 | 开放集拒识 / 混合物解混 | ⬜ |
| P5 | 工程化：CLI、单元测试、文档 | ⬜ |

## 环境

- Python 3.13
- numpy 2.5 / scipy 1.18 / matplotlib 3.11
- pybaselines 1.2.x

## 参考与致谢

- [pybaselines](https://github.com/derb12/pybaselines) —— 基线校正算法库，BSD-3-Clause
- ModPoly / IModPoly 原始方法出自 Lieber & Mahadevan-Jansen (2003)

## 许可

本项目代码采用 MIT License，见 [LICENSE](LICENSE)。
