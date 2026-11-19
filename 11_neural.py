"""P8：非线性表征学习能否突破线性基线。

背景
    P2–P7 的六轮尝试都建立在"线性手段"上（线性 SVM、线性变换、距离度量），
    全部未突破 SNV + 全维 + 线性 SVM 的 0.828。本轮换非线性模型。

两种模型
    mlp   sklearn 的 MLP 分类器（1 个隐藏层）。纯非线性基线，
          不需要 PyTorch，用来回答"非线性本身能不能带来提升"。
    cnn   一维卷积网络。卷积核在波数方向滑动，天然对应"局部峰形"这一物理先验，
          是比 MLP 更贴合光谱结构的归纳偏置。

    cnn 还支持 --mixstyle：在训练时按概率交换不同批次样本的特征统计量，
    迫使模型不依赖"某个批次的整体风格"。这是域泛化里最轻量的一种做法。

评估
    仍然只认留一批次协议。随机划分那一列仅作参考。

用法
    .venv/Scripts/python.exe 11_neural.py --model mlp
    .venv/Scripts/python.exe 11_neural.py --model cnn --mixstyle
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from raman_rec import load_bacterial, load_oil, stratified_split
from raman_rec.classify import prepare_matrix
from raman_rec.datasets import Dataset
from raman_rec.transfer import row_snv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SVM_BASELINE = {"bacterial": 0.828, "oil": 0.898}


def make_folds(dataset: Dataset):
    _, _, groups = prepare_matrix(dataset, normalize=None)
    groups = np.asarray(groups)
    train_ds, test_ds = stratified_split(dataset, test_size=0.3, seed=0)
    id2idx = {id(s): i for i, s in enumerate(dataset.spectra)}
    tr = np.array([id2idx[id(s)] for s in train_ds.spectra])
    te = np.array([id2idx[id(s)] for s in test_ds.spectra])
    folds = []
    for g in sorted({g for g in groups if g}):
        test_i = np.where(groups == g)[0]
        train_i = np.where((groups != g) & (groups != ""))[0]
        if test_i.size and train_i.size:
            folds.append((g, train_i, test_i))
    return tr, te, folds, groups


# ---------------------------------------------------------------- MLP
def fit_predict_mlp(X_tr, y_tr, X_te, y_te, seed: int = 0,
                    hidden: int = 128) -> float:
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                        alpha=1e-3, batch_size=64, learning_rate_init=1e-3,
                        max_iter=200, early_stopping=True,
                        n_iter_no_change=15, random_state=seed)
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(accuracy_score(y_te, clf.predict(scaler.transform(X_te))))


# ---------------------------------------------------------------- CNN
def _require_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception as exc:                      # torch 可能被策略拦截
        print(f"  [跳过 CNN] torch 不可用：{type(exc).__name__}: {exc}")
        return False


def _build_cnn(n_classes: int, length: int):
    """一维卷积网络。

    卷积核宽度是调过的一个关键量：核 15（约 15 个波数点）时随机划分只有 0.69，
    核 31 时升到 0.81。拉曼峰半高宽常在 10–30 cm⁻¹，核太窄看不全峰形。
    """
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(128, n_classes)

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    return CNN()


def _mixstyle(feat, groups, p: float = 0.5, alpha: float = 0.1):
    """MixStyle：按概率把不同批次样本的特征均值/方差互换。

    迫使模型不去依赖"这一批的整体风格"，是域泛化里最轻量的一种正则。
    """
    import torch

    if feat.dim() != 3 or np.random.rand() > p:
        return feat
    mu = feat.mean(dim=2, keepdim=True)
    sig = feat.std(dim=2, keepdim=True) + 1e-6

    idx = torch.randperm(feat.size(0), device=feat.device)
    # 尽量只在不同批次之间交换
    mask = groups != groups[idx]
    if mask.sum() < 2:
        return feat
    lam = float(np.random.beta(alpha, alpha))
    mu_mix = mu * lam + mu[idx] * (1 - lam)
    sig_mix = sig * lam + sig[idx] * (1 - lam)
    out = (feat - mu) / sig * sig_mix + mu_mix
    return torch.where(mask.view(-1, 1, 1), out, feat)


def fit_predict_cnn(X_tr, y_tr, g_tr, X_te, y_te, seed: int = 0,
                    epochs: int = 80, mixstyle: bool = False) -> float:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    classes = np.unique(y_tr)
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls2idx[c] for c in y_tr])

    # 逐特征标准化，再转成 (N, 1, L)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    Xtr = torch.tensor((X_tr - mu) / sd, dtype=torch.float32).unsqueeze(1)
    Xte = torch.tensor((X_te - mu) / sd, dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(y_idx, dtype=torch.long)
    g_codes = {g: i for i, g in enumerate(np.unique(g_tr))}
    gtr = torch.tensor([g_codes[g] for g in g_tr])

    model = _build_cnn(len(classes), Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(Xtr, ytr, gtr)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
                                         drop_last=True)

    model.train()
    for _ in range(epochs):
        for xb, yb, gb in loader:
            if mixstyle:
                # 在第一个卷积块之后插入 MixStyle
                f = model.features[0](xb)
                f = _mixstyle(f, gb)
                for layer in list(model.features)[1:]:
                    f = layer(f)
                logits = model.head(f.flatten(1))
            else:
                logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1).cpu().numpy()
    return float(np.mean(classes[pred] == y_te))


# ---------------------------------------------------------------- 评估
def evaluate(dataset: Dataset, model_kind: str, mixstyle: bool,
             max_folds: int | None = None, epochs: int = 60) -> dict:
    X, y, groups = prepare_matrix(dataset, normalize=None)
    X = row_snv(X)
    tr_idx, te_idx, folds, groups = make_folds(dataset)
    if max_folds:
        folds = folds[:max_folds]

    def run(train_i, test_i):
        if model_kind == "mlp":
            return fit_predict_mlp(X[train_i], y[train_i], X[test_i], y[test_i])
        return fit_predict_cnn(X[train_i], y[train_i], groups[train_i],
                               X[test_i], y[test_i], mixstyle=mixstyle,
                               epochs=epochs)

    acc_rand = run(tr_idx, te_idx)
    accs = []
    for name, train_i, test_i in folds:
        acc = run(train_i, test_i)
        accs.append(acc)
        print(f"    留出 {name:>12s}  准确率 {acc:.3f}")

    return {"random": acc_rand, "loo_mean": float(np.mean(accs)),
            "loo_std": float(np.std(accs)), "n_folds": len(accs)}


def plot_results(results: dict[tuple[str, str], dict], out_path: Path) -> None:
    keys = list(results)
    fig, ax = plt.subplots(figsize=(max(7.0, 2.2 * len(keys)), 5.4))
    x = np.arange(len(keys))
    width = 0.36

    base = [SVM_BASELINE[k[0]] for k in keys]
    loo = [results[k]["loo_mean"] for k in keys]
    err = [results[k]["loo_std"] for k in keys]

    ax.bar(x - width / 2, base, width, label="线性 SVM 基线", color="#B0AFA8")
    ax.bar(x + width / 2, loo, width, yerr=err, capsize=3,
           label="非线性模型", color="#4C78A8",
           error_kw={"lw": 1, "ecolor": "#333333"})

    for xi, k in enumerate(keys):
        ax.text(xi - width / 2, base[xi] + 0.012, f"{base[xi]:.3f}",
                ha="center", fontsize=8)
        # 数值标在误差棒上端，避免与误差棒重叠
        top = loo[xi] + err[xi] + 0.015
        ax.text(xi + width / 2, top, f"{loo[xi]:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("留一批次准确率", fontsize=10)
    ax.set_title("非线性模型 vs 线性基线（误差棒为各折标准差）", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print("图已保存:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="非线性模型实验（MLP / 1D-CNN）")
    parser.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="bacterial")
    parser.add_argument("--model", choices=["mlp", "cnn", "both"], default="mlp")
    parser.add_argument("--mixstyle", action="store_true",
                        help="CNN 训练时启用 MixStyle 域泛化")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="只跑前 N 折，用于快速验证")
    parser.add_argument("--epochs", type=int, default=80,
                        help="CNN 训练轮数（30 轮会明显欠拟合）")
    parser.add_argument("--append", action="store_true",
                        help="把本次结果并入已有的 neural_result.csv 再统一绘图"
                             "（用于把多次运行合成一张对比图）")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    kinds = ["mlp", "cnn"] if args.model == "both" else [args.model]
    if "cnn" in kinds and not _require_torch():
        kinds = [k for k in kinds if k != "cnn"]
        if not kinds:
            print("torch 不可用，且未选择 mlp。改用 --model mlp 运行。")
            return 1

    results: dict[tuple[str, str], dict] = {}
    for ds_name in (["bacterial"] if args.dataset == "bacterial"
                    else ["oil"] if args.dataset == "oil" else ["bacterial", "oil"]):
        print("=" * 72)
        print(f"数据集：{ds_name}")
        ds = (load_bacterial(DATA_DIR / "Bacterial_data") if ds_name == "bacterial"
              else load_oil(DATA_DIR / "Oil_data"))
        for kind in kinds:
            label = {"mlp": "MLP", "cnn": "1D-CNN"}[kind]
            if kind == "cnn" and args.mixstyle:
                label += "+MixStyle"
            print(f"  [{label}]")
            r = evaluate(ds, kind, args.mixstyle, args.max_folds, args.epochs)
            results[(ds_name, label)] = r
            print(f"  → 随机 {r['random']:.3f} | 留批次 "
                  f"{r['loo_mean']:.3f} ± {r['loo_std']:.3f}")

    if results:
        csv_path = ROOT / "neural_result.csv"
        if args.append and csv_path.exists():
            import csv as _csv
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in _csv.DictReader(f):
                    key = (row["dataset"], row["model"])
                    if key not in results:      # 本次结果优先
                        results[key] = {
                            "random": float(row["random_acc"]),
                            "loo_mean": float(row["loo_mean"]),
                            "loo_std": float(row["loo_std"]),
                            "n_folds": int(row["n_folds"]),
                        }
            results = dict(sorted(results.items()))

        plot_results(results, ROOT / "neural_result.png")
        lines = ["dataset,model,random_acc,loo_mean,loo_std,n_folds"]
        for (ds_name, label), r in results.items():
            lines.append(f"{ds_name},{label},{r['random']:.4f},"
                         f"{r['loo_mean']:.4f},{r['loo_std']:.4f},{r['n_folds']}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        print("表已保存:", csv_path)

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
