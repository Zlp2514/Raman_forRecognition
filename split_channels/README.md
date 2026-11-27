# split_channels —— 把一条拉曼谱拆成「荧光通道」与「拉曼通道」

从一条真实光谱出发，同时产出两条曲线并分别成图：

```text
原谱 y(ν) ──► 荧光通道 F(ν)  （缓慢变化的宽背景）
          └─► 拉曼通道 R(ν) = y(ν) − F(ν)  （窄峰）
```

**这是数学估计，不是物理分离。** F(ν) 是"形状上能替代荧光的曲线"，不等于样品真实发射的荧光光谱。
需要真荧光请用 SERDS 双波长差谱或时间门控（见文末）。

---

## 快速开始

> 下面的命令假设你在**仓库根目录**（含 `README.md` 的那一层）执行。
> 如果你在上一级工作区目录，把 `split_channels/` 换成 `raman_reproduction/split_channels/` 即可。
>
> **不带任何参数运行也可以**：默认处理内置的 3 条示例谱，并弹出图窗。
> 只存图不弹窗，加 `--no-show`；看参数说明用 `--help`。

```bash
# 1. 生成 3 条合成示例谱并处理（不需要任何数据）
python split_channels/01_split_two_channels.py --demo

# 2. 处理自己的文件或整个文件夹
python split_channels/01_split_two_channels.py --input my_raman.csv
python split_channels/01_split_two_channels.py --input ./my_data --method asls

# 3. 对比四种方法给出的荧光曲线差异
python split_channels/01_split_two_channels.py --input split_channels/demo/spectrum_01.csv --compare
```

从工作区根目录（仓库的上一级）运行时，写成完整前缀：

```bash
.\.venv\Scripts\python.exe .\raman_reproduction\split_channels\01_split_two_channels.py --demo
```

输入格式：两列 CSV，第一行表头，第一列位移（cm⁻¹），第二列强度。

---

## 输出

| 文件 | 内容 |
|---|---|
| `<名字>_channels.csv` | 四列：`raman_shift, intensity, fluorescence, raman` |
| `<名字>_channels.png` | 三联图：原谱+荧光 / 荧光通道 / 拉曼通道 |
| `qc_summary.csv` | 每条谱的质检指标与结论（每次运行重写） |
| `compare_<名字>.png` | `--compare` 时生成的方法对比图 |

图中底部一行是质检摘要：过冲、峰位最大偏移、峰面积保留率、峰数。

---

## 用自己填的数据：my_ramantest.csv

`my_ramantest.csv` 是一份可直接编辑的模板（361 点、200–2000 cm⁻¹，含 4 个模拟峰）。
把它换成自己的数据，然后运行：

```bash
python split_channels/02_process_my_ramantest.py
```

输出（与脚本同目录）：

| 文件 | 内容 |
|---|---|
| `my_ramantest_channels.csv` | 四列：`raman_shift, intensity, fluorescence, raman` |
| `my_ramantest_result.png` | 三联图：原谱+荧光 / 荧光通道 / 拉曼通道 |

常用参数：

```bash
python split_channels/02_process_my_ramantest.py --order 6          # 指定阶数
python split_channels/02_process_my_ramantest.py --method asls      # 换方法
python split_channels/02_process_my_ramantest.py --input 别的.csv    # 换输入文件
python split_channels/02_process_my_ramantest.py --no-show          # 只存图
```

**点数不足时脚本会自动降阶**：≥80 点用 5 阶；更少时按 `点数/8` 降阶并打印提示，
避免出现"6 个点配 5 阶多项式"那种把数据解释完的假结果。

填数据的三条要求：第一列位移从小到大不重复、第二列填**原始 counts**（不要先扣基线或归一化）、
至少 12 个点（建议 300 点以上）。

---

## 六种方法

| 方法 | 参数 | 假设 | 适用 |
|---|---|---|---|
| `imodpoly` | `--order`（默认 5） | 背景可用低阶多项式描述 | 通用默认，荧光平滑时最好 |
| `modpoly` | `--order` | 同上，但对噪声更敏感 | 高信噪比谱 |
| `asls` | `--lam --p` | 背景平滑 + 峰稀疏 | 荧光形状不像多项式时更稳 |
| `airpls` | `--lam` | 自动迭代重加权 | 免调参，光谱预处理常用 |
| `snip` | `--order`（窗口系数） | 峰比背景窄，可逐层削去 | 窄峰谱、能谱类数据 |
| `lowpass` | `--sigma` | 荧光是低频，滤波即得 | **仅作对照，见下方警告** |

### 关于 `lowpass` 的实测警告

`--compare` 的默认输出里能直接看到：`lowpass(sigma=25)` 的荧光曲线在峰位被"拽下去"，
对应的拉曼通道出现约 −40 的负坑，峰高被压掉三成。

原因很简单：高斯平滑的尺度（25 点 ≈ 50 cm⁻¹）和拉曼峰的宽度（10–16 cm⁻¹）是同一量级，
滤波分不清"宽峰"和"背景"。**只有当峰远窄于背景变化时，频域分离才成立**——这正是
Mosier-Boss et al. 1995 那篇 FFT 方法的前提。

---

## 五项质检（全部不需要真值）

| 指标 | 定义 | 判据 | 说明 |
|---|---|---|---|
| **负向过冲** | `|min(R)| / max(R)` | < 0.25 | 扣完后是零均值残差，"一半点为负"是正常的；要看最深的负坑相对主峰有多深 |
| **峰位最大偏移** | 拉曼峰与参考峰配对后的最大位移 | < 5 cm⁻¹ | 参考峰取自"原谱去线性趋势 + 轻度平滑"；只在 ±40 采样点内配对，避免跨峰误配 |
| **峰面积保留率** | 峰窗内 `∫R / ∫(y−弦)` | 0.8–1.2 | 低于 0.8 说明荧光曲线吃掉了拉曼峰 |
| **阶数敏感度** | `order±1` 时 F 的最大相对变化 | < 0.05 | 仅多项式法；偏大说明阶数在拟合噪声 |
| **粗糙度比** | `RMS(Δ²F) / RMS(Δ²R)` | < 0.5 | F 必须远比 R 平滑，否则分离没意义 |

结论只有 `通过` / `警告` 两档，警告会给出具体原因。

---

## 常见报错

**`PermissionError: [Errno 13] Permission denied: ...\demo\spectrum_01.csv`**

文件正被 Excel / WPS / VS Code 的表格预览占用，Windows 会加独占锁，任何程序都无法覆盖写入。

处理：关掉那个文件的预览窗口，或删掉 `demo/` 目录让脚本重建。脚本已做兼容——示例谱已存在时
不会重写，被占用时只打印一行警告并继续处理，不影响其余文件。

**运行后没有弹出图窗**

默认会弹窗（matplotlib 后端 TkAgg）。如果没看到窗口：可能被终端挡住了、或在远程/无显示环境下运行。
图片无论如何都会保存到 `output/`，直接打开 PNG 即可。批量跑或不想弹窗时加 `--no-show`。

---

## 三条使用边界

1. **先分离，再归一化。** SNV 会逐条减均值除标准差，恰好把荧光幅度信息抹掉；要保留荧光通道，
   必须在 SNV 之前导出。
2. **不要对已扣过基线的数据再跑一次。** 第二次拿到的是残差，不是荧光。
3. **宽拉曼峰会被当成荧光。** 判断方法：`--order` 取 3 / 5 / 7 各跑一次，看"峰面积保留率"
   是否随阶数剧烈变化；剧烈变化说明峰正在被多项式吸收。

---

## 需要"真分离"时

| 手段 | 论文 | 代价 |
|---|---|---|
| SERDS（移位激发差谱） | Shreve, Cherepy, Mathies, *Appl. Spectrosc.* 46(4):707–711, 1992 | 需要双波长激光 |
| 时间门控 | Kögler & Heilala, *Meas. Sci. Technol.* 32:012002, 2020 | 需要脉冲激光 + 门控探测器 |
| MCR-ALS 解混 | Tauler 等；*Microchem. J.* 2025 的 MCR-DA 工作 | 需要一组谱，不是单条 |

---

## 与本仓库其它脚本的关系

- `compare_baseline_methods.py`：比较六种**基线方法**在合成谱上的精度（有真值）
- `split_channels/01_split_two_channels.py`：把**任意真实谱**拆成两条可导出的曲线（无真值，靠质检）
- `02_process_my_csv.py`：只做 IModPoly 扣除，不单独输出荧光通道

两者共用 pybaselines，但目的不同：前者回答"哪个方法准"，后者回答"这条谱的荧光长什么样"。
