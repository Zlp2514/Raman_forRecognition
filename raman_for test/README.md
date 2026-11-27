# raman_for test · 03–22 号实验（脚本与产出）

03–22 号脚本及其产出**已从仓库根目录迁移到这里**，并按「谁写的」分成两个子目录。
根目录现在只剩入口脚本（01/02）、六方法对比脚本、`run.py` 与核心包 `raman_rec`。

---

## 目录结构

```
raman_for test/
├── README.md                    ← 本文件
├── 01_我自己写的_03-11/          9 个脚本 + 17 个产出（11/19 12:15–18:54）
└── 02_AI协助_12-22/             11 个脚本 + 28 个产出（11/19 19:30–22:05）
```

## 怎么运行

脚本已迁到子目录，但仍需 `raman_rec` 包与 `data/`，所以**在仓库根目录**用统一入口调用：

```bash
cd ..                                     # 回到 raman_reproduction/
.venv/Scripts/python.exe run.py 21 --no-show
.venv/Scripts/python.exe run.py 12 --stage apply --method asls
```

`run.py` 会按编号找到对应脚本，并把仓库根目录注入模块搜索路径。

也可以直接指定路径（同样在根目录执行）：

```bash
.venv/Scripts/python.exe "raman_for test/01_我自己写的_03-11/03_load_dataset.py"
```

**不要 `cd` 进子目录再运行**——那样 `raman_rec` 找不到，`data/` 也找不到。

### 每个脚本的改动

迁移时在文件头插入了三行路径引导，其余逻辑未改：

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

数据路径改为 `DATA_DIR = REPO_ROOT / "data"`；产出仍写在**脚本所在目录**，
所以每个子目录是自包含的（脚本 + 它的图 + 它的表）。

`18_improvement_summary.py` 额外加了一层输入查找：它要读 03–11 生成的
`dimension_sweep.csv`、`augmentation_result.csv` 等，会按
「本目录 → 同级 01 目录 → 仓库根」的顺序寻找。

---

## 一、你自己写的（03–11）

| 脚本 | 产出 | 做了什么 |
|---|---|---|
| 03_load_dataset.py | dataset_overview.png | 两个数据集的类别分布与平均谱，确认数据形态 |
| 04_classify.py | classification_result.png / .csv | 基线分类：SNV + 全维 + 线性 SVM，双协议对照 |
| 05_batch_effect.py | batch_effect_result.png / .csv | 浅层校正（导数、波段、批次中心化、CORAL）对比 |
| 06_augmentation.py | augmentation_result.png / .csv | 训练期数据增强（随机扰动、背景形状、波数平移） |
| 07_prototype.py | prototype_result.png / .csv | 分类器替换：原型（欧氏/余弦/白化）与 kNN |
| 08_dimension_sweep.py | dimension_sweep.png / .csv | 维度扫描：PCA 20→1376 对准确率的影响 |
| 09_metric_learning.py | metric_learning_result.png / .csv | 线性度量学习：LDA（含收缩）+ SVM / kNN / 原型 |
| 10_openset.py | openset_result.png / .csv | 开放集：margin / 原型距离 / 马氏距离三种评分器 |
| 11_neural.py | neural_result.png / .csv | 非线性模型：MLP 与一维 CNN |

## 二、AI 协助的（12–22）

| 脚本 | 产出 | 做了什么 |
|---|---|---|
| 12_param_selection.py | param_selection_validate.*、param_selection_apply_asls.csv、param_selection_apply_imodpoly.csv | 无真值代理指标选参；σ=20 噪声鲁棒性；两种方法的下游验证 |
| 13_library_match.py | library_match_result_raw / _deriv | 谱库匹配最朴素实现：相关 + 全局阈值 |
| 14_library_match_v2.py | library_v2_result_bacterial / _oil | 改进版：峰位对齐 + Fisher 加权 + 分类别阈值 |
| 15_cascade.py | cascade_result.* | 级联：库匹配门控 + SVM 判定，三方案对比 |
| 16_seed_stability.py | seed_stability_bacterial.* | 5 个随机种子 vs 折间波动 |
| 17_attribution.py | attribution_result.* | 判别权重峰位 + 波段置换重要性 |
| 18_improvement_summary.py | improvement_summary.* | 六类改进汇总（19 个配置统一到全维基线） |
| 19_paired_test.py | paired_test.* | 9 折 Wilcoxon 符号秩配对检验 |
| 20_openset_perclass.py | openset_perclass.* | 分类别拒识能力（AUROC 0.10–0.99） |
| 21_protocol_gap.py | protocol_gap.* | 双协议差距与逐批次画像 |
| 22_rejection_tradeoff.py | rejection_ceiling.png、rejection_tradeoff.csv | 拒识净收益扫描与 1/N 上限 |

## 三、归属依据

按文件修改时间划分，不是凭回忆：

- **03–11**：11/19 12:15–18:54，两小时内连续完成
- **12–22**：11/19 19:30–22:05，AI 接手的那一段（12 号是在原有选参思路上重写的验证部分）

哪条记得是自己写的，告知即可调整。

## 四、仍留在根目录的文件

| 文件 | 原因 |
|---|---|
| `01_quickstart.py` / `02_process_my_csv.py` | 入门入口，不算 03–22 范围 |
| `compare_baseline_methods.py`、`compare_result.png` | 六方法横向对比，更早的独立脚本（PPT 第 8 页的图来自它） |
| `cli.py`、`run.py` | 命令行入口与统一运行入口 |
| `demo_raman*` / `my_raman*` | 01/02 的演示与自备数据 |
| `png_demo_*.png` | 环境自检临时图 |

## 五、环境提醒

这台机器开着 Windows「智能应用控制」，会拦截未签名的 scipy/sklearn 扩展模块
（`_reordering`、`_isfinite` 等），导入可能偶发失败，**重试通常能过**。
长期使用建议迁到 WSL，见根目录 README 的说明。
