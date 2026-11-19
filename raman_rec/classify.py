"""光谱分类：把拉曼光谱变成可分类的特征向量，并给出可信的评估。

为什么评估协议比模型更重要
    光谱分类最容易出的错不是"模型不够强"，而是"评估方式太乐观"。
    同一批数据如果按随机划分，测试集和训练集往往来自同一个测量批次、
    同一片载玻片、同一台仪器的同一时段，模型可能只是记住了那次测量的
    整体偏移，而没有学到物质的分子特征。
    因此本模块默认给出两种协议：

        random_split   随机分层划分（乐观，接近文献常见报告值）
        group_cv       留一组交叉验证（严格，检验跨批次泛化）

    两者的差距本身就是有价值的结论。

特征构造
    每条光谱 → （可选基线校正）→ 归一化 → 展平为向量
    再交给 StandardScaler → PCA → 分类器 的流水线。
    PCA 的作用是把上千维降到几十维，既降噪也避免小样本过拟合。
"""
from __future__ import annotations

import numpy as np

from .datasets import Dataset
from .preprocess import remove_baseline
from .spectrum import Spectrum


# ---------------------------------------------------------------- 特征矩阵
def spectrum_to_vector(
    spec: Spectrum,
    normalize: str | None = "snv",
    baseline: str | None = None,
    **baseline_kwargs,
) -> np.ndarray:
    """把一条光谱转成特征向量。

    参数
        normalize  归一化方式（见 Spectrum.normalized）；None 表示不归一化
        baseline   基线校正方法；None 表示数据已校正过，不再处理。
                   实测数据如果已由仪器软件扣过背景，重复扣除反而会引入负峰。
    """
    y = spec.y
    if baseline is not None:
        y = remove_baseline(spec, baseline, **baseline_kwargs).y
    if normalize is not None:
        y = Spectrum(x=spec.x, y=y).normalized(normalize).y
    return np.asarray(y, dtype=float)


def prepare_matrix(
    dataset: Dataset,
    normalize: str | None = "snv",
    baseline: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """把整个数据集转成 (X, y, groups)。

    返回
        X       形状 (样本数, 点数)
        y       标签数组
        groups  分组标签（如测量批次），没有则为空字符串
    """
    lengths = {len(s) for s in dataset.spectra}
    if len(lengths) != 1:
        raise ValueError(f"数据集中光谱点数不一致：{sorted(lengths)}，需先重采样到同一轴")

    X = np.vstack([spectrum_to_vector(s, normalize=normalize, baseline=baseline)
                   for s in dataset.spectra])
    y = np.array([s.label or "" for s in dataset.spectra])
    groups = [str(s.meta.get("batch") or "") for s in dataset.spectra]
    return X, y, groups


# ---------------------------------------------------------------- 模型
def build_pipeline(
    n_components: int = 20,
    classifier: str = "svc",
    random_state: int = 0,
):
    """构建 StandardScaler → PCA → 分类器 的流水线。

    classifier 可选
        svc     线性支持向量机，光谱分类上通常最稳
        lda     线性判别分析，自带降维，小样本友好
        logreg  多分类逻辑回归，输出概率便于做拒识（后续 P4 会用到）
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if classifier == "svc":
        clf = SVC(kernel="linear", C=1.0, random_state=random_state)
    elif classifier == "lda":
        clf = LinearDiscriminantAnalysis()
    elif classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, multi_class="auto",
                                 random_state=random_state)
    else:
        raise ValueError(f"未知分类器 {classifier!r}，可选 svc / lda / logreg")

    steps = [("scaler", StandardScaler())]
    if not isinstance(clf, LinearDiscriminantAnalysis) and n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ---------------------------------------------------------------- 评估
def metrics(y_true, y_pred, labels: list[str]) -> dict:
    """计算准确率、宏平均 F1 与混淆矩阵。"""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
        "n": int(len(y_true)),
    }


def random_split_eval(
    dataset: Dataset,
    test_size: float = 0.3,
    seed: int = 0,
    n_components: int = 20,
    classifier: str = "svc",
    normalize: str | None = "snv",
    baseline: str | None = None,
) -> dict:
    """随机分层划分评估（乐观协议）。"""
    from .datasets import stratified_split

    train, test = stratified_split(dataset, test_size=test_size, seed=seed)
    X_tr, y_tr, _ = prepare_matrix(train, normalize=normalize, baseline=baseline)
    X_te, y_te, _ = prepare_matrix(test, normalize=normalize, baseline=baseline)

    model = build_pipeline(n_components=n_components, classifier=classifier)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return metrics(y_te, pred, dataset.classes)


def group_cv_eval(
    dataset: Dataset,
    group_key: str = "batch",
    n_components: int = 20,
    classifier: str = "svc",
    normalize: str | None = "snv",
    baseline: str | None = None,
) -> dict:
    """留一组交叉验证（严格协议）。

    每一折：留出一个批次作为测试集，其余批次全部训练。
    返回每折准确率、平均与标准差，以及最后一折的混淆矩阵。
    """
    from .datasets import group_split

    folds = group_split(dataset, group_key=group_key)
    if not folds:
        raise ValueError(f"数据集没有可用的分组字段 {group_key!r}")

    labels = dataset.classes
    fold_acc: list[float] = []
    fold_names: list[str] = []
    last_cm = None
    for name, train, test in folds:
        X_tr, y_tr, _ = prepare_matrix(train, normalize=normalize, baseline=baseline)
        X_te, y_te, _ = prepare_matrix(test, normalize=normalize, baseline=baseline)
        model = build_pipeline(n_components=n_components, classifier=classifier)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        m = metrics(y_te, pred, labels)
        fold_acc.append(m["accuracy"])
        fold_names.append(name)
        last_cm = m["confusion"]

    return {
        "folds": fold_names,
        "fold_accuracy": fold_acc,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "labels": labels,
        "confusion_last": last_cm,
    }


# ---------------------------------------------------------------- 可视化
def plot_confusion(cm: np.ndarray, labels: list[str], ax, title: str = "") -> None:
    """在给定坐标轴上画混淆矩阵（行=真实，列=预测）。"""
    n = len(labels)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title, fontsize=11)

    total = cm.sum() or 1
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            value = cm[i, j]
            ax.text(j, i, f"{value}", ha="center", va="center", fontsize=8,
                    color="white" if value > thresh else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    _ = total