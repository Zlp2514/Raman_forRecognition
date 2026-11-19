"""谱库匹配：把"已知谱库 + 待测谱比对"这套工程思路做成可测量的最小实现。

为什么单独做这一层
    前面的实验都是在"闭集"里训练分类器——模型见过所有类别。而实际做分子识别时，
    更常见的做法是建一个**谱库**（每种物质存几条代表谱），待测谱进来后与库逐条比对，
    取最相似的那条作为答案；相似度太低就报"库中没有"。

    这条路线和训练分类器有本质区别：
      · 库可以随时增删物质，不需要重训模型；
      · 但它的准确性完全依赖"库谱与待测谱的测量条件是否一致"。

    本项目此前的实验已经量化了条件不一致的代价（留批次掉约 10 个百分点），
    本模块把它换算成"库匹配"语境下的数字：闭集准确率、库外拒识率、开放集准确率。

三个可调的设计量
    库规模      每个物质存几条代表谱（1 / 3 / 5 / 10）。库越大覆盖变异越多，但检索也越慢。
    相似度      Pearson 相关（谱库检索的常规做法，对整体强度不敏感）。
    拒识阈值    由库自身的"正确匹配得分分布"的下分位确定，不依赖任何外部真值。
"""
from __future__ import annotations

import numpy as np


def build_library(Z: np.ndarray, y: np.ndarray, per_class: int = 3,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """为每个类别挑 per_class 条代表谱。

    per_class=1 时取类均值（最省，但会削平个体差异）；
    per_class>1 时用**最远点采样**（k-center）：先取离类中心最近的一条，
    之后每次取"到已选集合距离最大"的那条。这样选出的代表谱覆盖类内变异，
    且不需要聚类库——用纯 numpy 实现，避免引入额外依赖与 DLL 风险。
    """
    lib_X, lib_y = [], []
    for c in np.unique(y):
        Zc = Z[y == c]
        k = int(min(per_class, len(Zc)))
        if k == 1:
            lib_X.append(Zc.mean(axis=0, keepdims=True))
            lib_y.append(np.array([c]))
            continue

        centroid = Zc.mean(axis=0, keepdims=True)
        d2 = ((Zc - centroid) ** 2).sum(axis=1)
        chosen = [int(np.argmin(d2))]                    # 从最靠近中心的样本开始
        min_d = ((Zc - Zc[chosen[0]]) ** 2).sum(axis=1)
        for _ in range(k - 1):
            nxt = int(np.argmax(min_d))
            chosen.append(nxt)
            min_d = np.minimum(min_d, ((Zc - Zc[nxt]) ** 2).sum(axis=1))
        lib_X.append(Zc[chosen])
        lib_y.append(np.repeat(c, len(chosen)))
    return np.vstack(lib_X), np.concatenate(lib_y)


def match(Zq: np.ndarray, lib_X: np.ndarray, lib_y: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """返回每条查询谱的最高相似度与对应的库类别。

    相似度用 Pearson 相关：先各自中心化再算内积/模长，
    这样对整体强度与偏移不敏感，只比较谱形——与谱库检索的常规做法一致。
    """
    A = Zq - Zq.mean(axis=1, keepdims=True)
    B = lib_X - lib_X.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    sim = A @ B.T                                  # (n_query, n_lib)

    idx = np.argmax(sim, axis=1)
    return sim[np.arange(len(idx)), idx], lib_y[idx]


def genuine_threshold(Z_train: np.ndarray, y_train: np.ndarray,
                      lib_X: np.ndarray, lib_y: np.ndarray,
                      q: float = 0.05) -> float:
    """用训练样本的"正确匹配得分"分布确定拒识阈值。

    做法：对每条训练样本，看它与库中自己那一类的最大相似度（这就是"真匹配"该有的分数），
    取 q 分位作为阈值。全程只用训练集与库本身，不需要任何外部真值，
    也不存在用测试集定阈值的泄漏问题。
    """
    out = []
    for c in np.unique(y_train):
        Zc = Z_train[y_train == c]
        mask = lib_y == c
        if not mask.any() or len(Zc) == 0:
            continue
        A = Zc - Zc.mean(axis=1, keepdims=True)
        B = lib_X[mask] - lib_X[mask].mean(axis=1, keepdims=True)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
        out.append((A @ B.T).max(axis=1))
    if not out:
        return 0.0
    return float(np.quantile(np.concatenate(out), q))
