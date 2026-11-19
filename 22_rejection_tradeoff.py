"""P19：论文图 4——拒识的净收益存在 1/N 的上限。

要传达的一句话
    级联方案（库匹配门控 + 分类器判定）能不能净赚，不由实现质量决定，
    而由一个算术约束决定：**未知类只占测试样本的 1/N，收益上限就是 1/N；
    误拒的代价却作用在 (N-1)/N 的样本上。**

推导
    设已知类占比 (N-1)/N，闭集准确率 c；门控的已知接受率 a，未知拒识率 r，则

        open = c * a * (N-1)/N + r * (1/N)
        Δ    = r/N - c * FRR * (N-1)/N       （FRR = 1 - a）

    两条硬结论：
      · Δ ≥ 0 要求 r ≥ c * (N-1) * FRR；
      · 完美门控（FRR=0, r=1）的收益上限 Δ = 1/N —— 细菌 16.7 pt，油脂 12.5 pt。

做法
    沿用 P12 级联设置，对每个（未知类 × 批次折）只算一次门控得分与闭集准确率，
    然后对阈值分位数做纯查表扫描（keep_rate 0.50→1.00，11 档）。

用法
    .venv/Scripts/python.exe 22_rejection_tradeoff.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import group_split
from raman_rec.library import best_shift_similarity, build_library, fisher_weights
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

MAX_SHIFT = 12
PER_CLASS = 3
KEEP_RATES = np.round(np.linspace(0.50, 1.00, 11), 3)
COLOR = {"bacterial": "#4C78A8", "oil": "#E45756"}
CN = {"bacterial": "细菌（N=6）", "oil": "油脂（N=8）"}


def _norm(A: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def collect_scores(ds) -> list[dict]:
    """每个（未知类 × 批次折）收集门控得分与闭集准确率。

    已知测试样本的 (真标签, 预测类, 得分)、未知样本的 (预测类, 得分)、
    各类"真匹配得分"分布全部存原始值；阈值扫描是纯查表。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y, groups = prepare_matrix(ds, normalize=None)
    X = row_snv(X)
    groups = np.asarray(groups)
    id2idx = {id(sp): i for i, sp in enumerate(ds.spectra)}

    folds = []
    for fname, train, test in group_split(ds, group_key="batch"):
        tr = np.array([id2idx[id(sp)] for sp in train.spectra])
        te = np.array([id2idx[id(sp)] for sp in test.spectra])
        folds.append((fname, tr, te))

    out = []
    for unknown in np.unique(y):
        for fname, tr, te in folds:
            known_all = te[y[te] != unknown]
            unk = te[y[te] == unknown]
            lib_tr = tr[y[tr] != unknown]
            if lib_tr.size < 50 or known_all.size < 5 or unk.size < 3:
                continue

            w = np.sqrt(fisher_weights(X[lib_tr], y[lib_tr]))
            lib_X, lib_y = build_library(X[lib_tr] * w, y[lib_tr],
                                         per_class=PER_CLASS)

            def scores(Q):
                sim = best_shift_similarity(Q * w, lib_X, MAX_SHIFT)
                idx = np.argmax(sim, axis=1)
                return sim[np.arange(len(idx)), idx], lib_y[idx]

            s_known, p_known = scores(X[known_all])
            s_unk, p_unk = scores(X[unk])

            genuine = {}
            for c in np.unique(y[lib_tr]):
                Zc = X[lib_tr][y[lib_tr] == c]
                mask = lib_y == c
                if not mask.any() or len(Zc) == 0:
                    continue
                # lib_X 已是加权后的数据，这里不能再乘一次 w
                sim = _norm(Zc * w) @ _norm(lib_X[mask]).T
                genuine[str(c)] = sim.max(axis=1)

            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", SVC(kernel="linear", C=1.0, random_state=0))])
            pipe.fit(X[lib_tr], y[lib_tr])
            pred = pipe.predict(X[known_all])

            out.append({
                "unknown": str(unknown), "fold": fname,
                "n_classes": int(len(np.unique(y))),
                "y_known": y[known_all], "pred_known": pred,
                "p_known": p_known,
                "s_known": s_known, "p_unk": p_unk, "s_unk": s_unk,
                "genuine": genuine,
                "closed": float(np.mean(pred == y[known_all])),
            })
    return out


def sweep(items: list[dict], keep_rates) -> list[dict]:
    """对阈值分位数纯查表扫描，返回每档的 FRR / 拒识率 / 净收益（百分点）。"""
    frrs, rrs, gains = [], [], []
    for q in keep_rates:
        for it in items:
            N = it["n_classes"]
            tau = {c: float(np.quantile(v, 1 - q))
                   for c, v in it["genuine"].items()}
            thr_k = np.array([tau.get(str(c), -np.inf) for c in it["p_known"]])
            acc_k = it["s_known"] >= thr_k
            a = float(np.mean(acc_k))
            frr = 1 - a
            thr_u = np.array([tau.get(str(c), -np.inf) for c in it["p_unk"]])
            r = float(np.mean(it["s_unk"] < thr_u))

            correct_known = float(np.mean((it["pred_known"] == it["y_known"]) & acc_k))
            open_acc = correct_known * (N - 1) / N + r / N
            baseline_open = it["closed"] * (N - 1) / N
            gains.append((open_acc - baseline_open) * 100)

            frrs.append(frr)
            rrs.append(r)
    # frrs/rrs/gains 按 q 外层、条目内层累积，需 reshape 成 (档数, 条目数) 再按档平均
    n_it = len(items)
    mat = lambda v: np.array(v, dtype=float).reshape(len(keep_rates), n_it)
    return {"keep": [float(q) for q in keep_rates],
            "frr": mat(frrs).mean(axis=1).tolist(),
            "rr": mat(rrs).mean(axis=1).tolist(),
            "gain": mat(gains).mean(axis=1).tolist()}



def plot(results: dict[str, list[dict]], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    ax = ax1
    for name in names_plot(results):
        r = results[name]
        N = 6 if name == "bacterial" else 8
        ax.plot(r["frr"], r["rr"], "o-", lw=1.8, ms=5, color=COLOR[name],
                label=CN[name])
        frr_grid = np.linspace(0, max(r["frr"]) * 1.1, 50)
        ax.plot(frr_grid, 0.9 * (N - 1) * frr_grid, "--", lw=1.2,
                color=COLOR[name], alpha=0.65, label="盈亏平衡线（c≈0.9）")
    ax.plot([0], [1], marker="*", ms=16, color="#333333", ls="")
    ax.annotate("完美门控", (0.012, 0.985), fontsize=8.5, ha="left")
    ax.set_xlabel("已知类误拒率（FRR）")
    ax.set_ylabel("未知类拒识率")
    ax.set_xlim(-0.02, None)
    ax.set_ylim(0, 1.05)
    ax.set_title("门控权衡：点在盈亏线上方才有净收益", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax = ax2
    for name in names_plot(results):
        r = results[name]
        N = 6 if name == "bacterial" else 8
        ax.plot(r["keep"], [g for g in r["gain"]], "o-", lw=1.8, ms=5,
                color=COLOR[name], label=CN[name])
    ax.axhline(0, color="#333333", lw=1.1)
    for N, color in ((6, "#4C78A8"), (8, "#E45756")):
        ax.axhline(100 / N, ls="--", lw=1.1, color=color, alpha=0.7)
        ax.text(1.004, 100 / N + 0.4, f"1/N = {100/N:.1f} pt", fontsize=8,
                va="bottom", ha="right", color=color)
    ax.set_xlabel("keep_rate（已知类目标接受率，越大阈值越松）")
    ax.set_ylabel("开放集净收益（百分点）")
    ax.set_title("净收益被 1/N 上限卡住，且随阈值收紧迅速转负", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def names_plot(results):
    return list(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="拒识收益上限")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"], default="both")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    names = ["bacterial", "oil"] if args.dataset == "both" else [args.dataset]
    results: dict[str, list[dict]] = {}
    lines = ["dataset,keep_rate,frr,reject_rate,gain_pt"]
    for name in names:
        ds = (load_bacterial(DATA_DIR / "Bacterial_data") if name == "bacterial"
              else load_oil(DATA_DIR / "Oil_data"))
        print(f"数据集：{name}（{len(ds)} 条，{len(ds.classes)} 类）")
        scores = collect_scores(ds)
        print(f"  收集 {len(scores)} 个（未知类 × 折）组合")
        r = results[name] = sweep(scores, KEEP_RATES)
        for i, q in enumerate(r["keep"]):
            lines.append(f"{name},{q:.3f},{r['frr'][i]:.4f},{r['rr'][i]:.4f},"
                         f"{r['gain'][i]:+.2f}")
        bi = int(np.argmax(r["gain"]))
        print(f"  最优档 keep={r['keep'][bi]:.2f}：FRR {r['frr'][bi]:.3f}、"
              f"拒识 {r['rr'][bi]:.3f}、净收益 {r['gain'][bi]:+.2f} pt")

    plot(results, ROOT / "rejection_ceiling.png")
    (ROOT / "rejection_tradeoff.csv").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8-sig")
    print("表已保存:", ROOT / "rejection_tradeoff.csv")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
