"""无真值选参：先验证代理指标可不可信。

两个阶段
    validate  在合成数据上（真值已知）检验"代理指标排序"与"真实误差排序"是否一致。
              这是整件事的前提：若秩相关很低，无真值选参就没有依据，后续不必做。
    apply     在真实光谱上按代理指标选参，比较"默认参数 / 代理选参 / 真实最优"三种做法的效果。

用法
    .venv/Scripts/python.exe 12_param_selection.py            # 默认跑 validate
    .venv/Scripts/python.exe 12_param_selection.py --stage apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import make_synthetic
from raman_rec.paramselect import proxy_metrics, score_candidates
from raman_rec.preprocess import PARAM_GRIDS, estimate_baseline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

METRIC_CN = {
    "neg_ratio": "负值比例",
    "baseline_curvature": "基线曲率",
    "lowfreq_residual": "低频残差",
    "noise_deviation": "噪声偏离",
    "composite": "综合得分",
}
METRICS = ["neg_ratio", "baseline_curvature", "lowfreq_residual", "noise_deviation"]


def evaluate_grid(spec, method: str) -> list[dict]:
    """对一个方法的参数网格逐个校正，记录真实误差与代理指标。"""
    true_baseline = spec.meta["true_baseline"]
    rows = []
    for params in PARAM_GRIDS[method]:
        try:
            baseline, info = estimate_baseline(spec.x, spec.y, method, **params)
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean((baseline - true_baseline) ** 2)))
        m = proxy_metrics(spec.y, baseline)
        rows.append({"method": method, "params": info["params"], "rmse": rmse, **m})
    return rows


def spearman(a, b) -> float:
    """Spearman 秩相关（无 scipy 依赖时也能算）。"""
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def run_validate(no_show: bool) -> int:
    spec = make_synthetic()
    print(f"合成光谱：{len(spec)} 点，噪声 σ = {spec.meta['noise_std']:.2f}")

    all_rows: list[dict] = []
    per_method: dict[str, list[dict]] = {}
    for method in PARAM_GRIDS:
        rows = evaluate_grid(spec, method)
        if len(rows) < 3:
            continue
        # 组内归一化后合成总分
        scores = score_candidates([{k: r[k] for k in METRICS} for r in rows])
        for r, s in zip(rows, scores):
            r["composite"] = float(s)
        per_method[method] = rows
        all_rows.extend(rows)

    # 合并后再算一次总分（跨方法可比），用于总体相关性
    pooled_scores = score_candidates([{k: r[k] for k in METRICS} for r in all_rows])
    for r, s in zip(all_rows, pooled_scores):
        r["composite_pooled"] = float(s)

    print("=" * 78)
    print(f"{'方法':<16}{'候选数':>6}{'负值比例':>10}{'基线曲率':>10}"
          f"{'低频残差':>10}{'信噪比':>10}{'综合':>10}")
    print("-" * 78)
    summary = []
    for method, rows in per_method.items():
        rmse = [r["rmse"] for r in rows]
        rho_comp = spearman(rmse, [r["composite"] for r in rows])
        summary.append((method, len(rows), rmse, rows, rho_comp))
        print(f"{method:<16}{len(rows):>6}"
              + "".join(f"{spearman(rmse, [r[k] for r in rows]):>10.2f}"
                        for k in METRICS)
              + f"{rho_comp:>10.2f}")

    print("-" * 78)
    pooled_rho = spearman([r["rmse"] for r in all_rows],
                          [r["composite_pooled"] for r in all_rows])
    print(f"{'全部候选合并':<16}{len(all_rows):>6}"
          + "".join(f"{spearman([r['rmse'] for r in all_rows], [r[k] for r in all_rows]):>10.2f}"
                    for k in METRICS)
          + f"{pooled_rho:>10.2f}")
    print("=" * 78)
    print("说明：前四项指标均为越小越好，故应与 RMSE 正相关；")
    print("      综合得分是越大越好，故应与 RMSE 负相关。")

    # 选参质量：按综合得分选出的参数，其真实误差在网格中排第几
    print("-" * 78)
    print(f"{'方法':<16}{'默认参数RMSE':>14}{'代理选参RMSE':>14}"
          f"{'真实最优RMSE':>14}{'选中排名':>10}")
    for method, _n, _rmse_list, rows, _rho in summary:
        best_proxy = max(rows, key=lambda r: r["composite"])
        best_true = min(rows, key=lambda r: r["rmse"])
        default_params = PARAM_GRIDS[method][0]
        default_row = next((r for r in rows if r["params"] == default_params), rows[0])
        rank = sorted(rows, key=lambda r: r["rmse"]).index(best_proxy) + 1
        print(f"{method:<16}{default_row['rmse']:>14.3f}{best_proxy['rmse']:>14.3f}"
              f"{best_true['rmse']:>14.3f}{rank:>7}/{len(rows)}")

    # 落盘
    lines = ["method,params,rmse,neg_ratio,baseline_curvature,lowfreq_residual,noise_deviation,composite"]
    for r in all_rows:
        p = ";".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(f"{r['method']},\"{p}\",{r['rmse']:.4f},{r['neg_ratio']:.6f},"
                     f"{r['baseline_curvature']:.6f},{r['lowfreq_residual']:.6f},"
                     f"{r['noise_deviation']:.6f},{r['composite']:.4f}")
    (ROOT / "param_selection_validate.csv").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8-sig")
    print("表已保存:", ROOT / "param_selection_validate.csv")

    # 图：左为各方法秩相关，右为综合得分 vs 真实 RMSE 散点
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    methods = [m for m, *_ in summary]
    x = np.arange(len(methods))
    width = 0.16
    for i, k in enumerate(METRICS):
        vals = [spearman(rm, [r[k] for r in rw]) for _, _, rm, rw, _ in summary]
        axes[0].bar(x + (i - 2) * width, vals, width, label=METRIC_CN[k])
    axes[0].axhline(0, color="#555555", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right", fontsize=8.5)
    axes[0].set_ylabel("与真实 RMSE 的 Spearman 秩相关")
    axes[0].set_title("各代理指标与真实误差的一致性", fontsize=11)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].scatter([r["rmse"] for r in all_rows], [r["composite_pooled"] for r in all_rows],
                    s=22, c="#4C78A8", alpha=0.75)
    axes[1].set_xlabel("真实基线误差 RMSE（越小越好）")
    axes[1].set_ylabel("综合代理得分（越大越好）")
    axes[1].set_title(f"综合得分 vs 真实误差（全部候选，ρ = {pooled_rho:.2f}）",
                      fontsize=11)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ROOT / "param_selection_validate.png", dpi=170, bbox_inches="tight")
    print("图已保存:", ROOT / "param_selection_validate.png")
    if not no_show:
        plt.show()
    return 0


def run_apply(no_show: bool, n_calib: int = 60, method: str = "imodpoly") -> int:
    """真实数据上的应用：代理选参能否改善下游识别。

    做法
        1. 用默认参数校正全部光谱，得到基线方案 A；
        2. 抽 n_calib 条光谱做"校准集"，对每条按代理指标选参，取多数票得到一套全局参数；
        3. 用该参数校正全部光谱，得到方案 B；
        4. 两套校正结果分别送进同一分类流水线（SNV → 全维 → 线性 SVM），
           比较留一批次准确率。

    真实数据没有真值，因此只能看下游任务是否受益——这也是选参最终的目的。
    """
    from raman_rec import load_bacterial
    from raman_rec.classify import prepare_matrix
    from raman_rec.transfer import row_snv
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    ds = load_bacterial(DATA_DIR / "Bacterial_data")
    X, y, groups = prepare_matrix(ds, normalize=None)
    groups = np.asarray(groups)
    print(f"真实数据：{len(ds)} 条光谱，{len(ds.classes)} 类")

    def correct_all(params: dict) -> np.ndarray:
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            b, _ = estimate_baseline(ds.spectra[i].x, X[i], method, **params)
            out[i] = X[i] - b
        return out

    # 步骤 1：默认参数
    default_params = dict(PARAM_GRIDS[method][0])
    print(f"默认参数：{default_params}")
    X_default = correct_all(default_params)

    # 步骤 2：在校准集上按代理指标选参，取多数票
    rng = np.random.default_rng(0)
    calib_idx = rng.choice(X.shape[0], size=min(n_calib, X.shape[0]), replace=False)
    votes = []
    for i in calib_idx:
        rows = []
        for params in PARAM_GRIDS[method]:
            b, info = estimate_baseline(ds.spectra[i].x, X[i], method, **params)
            rows.append({"params": info["params"],
                         **proxy_metrics(X[i], b)})
        scores = score_candidates([{k: r[k] for k in METRICS} for r in rows])
        votes.append(PARAM_GRIDS[method][int(np.argmax(scores))])

    keys = [tuple(sorted(v.items())) for v in votes]
    best_key = max(set(keys), key=keys.count)
    chosen = dict(best_key)
    agree = keys.count(best_key) / len(keys)
    print(f"代理选参：{chosen}（校准集 {len(calib_idx)} 条中 {agree:.0%} 的样本投给它）")
    X_chosen = correct_all(chosen)

    # 步骤 3：下游识别（留一批次）
    def loo_accuracy(Z: np.ndarray) -> tuple[float, float]:
        Zs = row_snv(Z)
        accs = []
        for g in sorted({g for g in groups if g}):
            te = np.where(groups == g)[0]
            tr = np.where((groups != g) & (groups != ""))[0]
            if te.size == 0 or tr.size == 0:
                continue
            clf = SVC(kernel="linear", C=1.0).fit(Zs[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(Zs[te])))
        return float(np.mean(accs)), float(np.std(accs))

    acc_d, std_d = loo_accuracy(X_default)
    acc_c, std_c = loo_accuracy(X_chosen)
    print("-" * 78)
    print(f"{'方案':<18}{'留批次准确率':>14}{'标准差':>10}")
    print(f"{'默认参数':<18}{acc_d:>14.3f}{std_d:>10.3f}")
    print(f"{'代理选参':<18}{acc_c:>14.3f}{std_c:>10.3f}")
    print(f"{'变化':<18}{acc_c - acc_d:>+14.3f}")
    print("-" * 78)
    print("说明：真实数据无真值，只能用下游识别准确率作为外部判据；")
    print("      准确率变化在 ±1 个标准差内应视为无差异。")

    lines = ["scheme,params,loo_accuracy,loo_std",
             f"default,\"{default_params}\",{acc_d:.4f},{std_d:.4f}",
             f"proxy_selected,\"{chosen}\",{acc_c:.4f},{std_c:.4f}"]
    (ROOT / "param_selection_apply.csv").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8-sig")
    print("表已保存:", ROOT / "param_selection_apply.csv")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="无真值基线参数选择")
    parser.add_argument("--stage", choices=["validate", "apply"], default="validate")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    if args.stage == "validate":
        return run_validate(args.no_show)
    return run_apply(args.no_show)


if __name__ == "__main__":
    raise SystemExit(main())
