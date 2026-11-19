"""核心模块的单元测试。

用标准库 unittest，不引入额外依赖：
    python -m unittest discover -s tests -v

覆盖范围
    Spectrum  构造校验、排序、归一化、寻峰、质量指标
    io        CSV 读写、编码容错、表头跳过
    preprocess 基线校正的形状与基本正确性
    augment   扩增后的样本数与形状
    prototype 在线性可分数据上应达到满准确率
    openset   评分接口、可分局面的 AUROC、阈值语义
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raman_rec import Spectrum, load_csv, save_csv, remove_baseline  # noqa: E402
from raman_rec.augment import augment  # noqa: E402
from raman_rec.openset import OpenSetModel, threshold_at  # noqa: E402
from raman_rec.prototype import NearestPrototype  # noqa: E402


def toy_spectrum(n: int = 200) -> Spectrum:
    """一条带两个高斯峰和缓慢背景的合成光谱。"""
    x = np.linspace(400, 1600, n)
    y = (50 * np.exp(-((x - 700) ** 2) / (2 * 12 ** 2))
         + 80 * np.exp(-((x - 1100) ** 2) / (2 * 15 ** 2))
         + 300 + 200 * np.exp(-(x - 400) / 500))
    return Spectrum(x=x, y=y)


class TestSpectrum(unittest.TestCase):
    def test_sorts_unsorted_input(self):
        spec = Spectrum(x=np.array([3.0, 1.0, 2.0]), y=np.array([30.0, 10.0, 20.0]))
        np.testing.assert_allclose(spec.x, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(spec.y, [10.0, 20.0, 30.0])

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            Spectrum(x=np.arange(5.0), y=np.arange(4.0))

    def test_rejects_duplicate_axis(self):
        with self.assertRaises(ValueError):
            Spectrum(x=np.array([1.0, 1.0, 2.0]), y=np.arange(3.0))

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            Spectrum(x=np.arange(3.0), y=np.array([1.0, np.nan, 3.0]))

    def test_normalized_modes(self):
        spec = toy_spectrum()
        snv = spec.normalized("snv")
        self.assertAlmostEqual(float(np.mean(snv.y)), 0.0, places=8)
        self.assertAlmostEqual(float(np.std(snv.y)), 1.0, places=8)

        mm = spec.normalized("minmax")
        self.assertAlmostEqual(float(np.min(mm.y)), 0.0, places=8)
        self.assertAlmostEqual(float(np.max(mm.y)), 1.0, places=8)

        with self.assertRaises(ValueError):
            spec.normalized("unknown-mode")

    def test_normalize_keeps_original_untouched(self):
        spec = toy_spectrum()
        before = spec.y.copy()
        spec.normalized("snv")
        np.testing.assert_allclose(spec.y, before)

    def test_peaks_finds_injected_peaks(self):
        spec = toy_spectrum()
        peaks = spec.peaks(prominence_frac=0.05, distance_cm=50)
        self.assertEqual(len(peaks["index"]), 2)
        np.testing.assert_allclose(np.sort(peaks["shift"]), [700, 1100], atol=30)

    def test_slice_limits_range(self):
        spec = toy_spectrum().slice(600, 1200)
        self.assertGreaterEqual(spec.x.min(), 600)
        self.assertLessEqual(spec.x.max(), 1200)

    def test_quality_metrics(self):
        spec = toy_spectrum()
        self.assertGreater(spec.snr(), 1.0)
        self.assertEqual(spec.negative_ratio(), 0.0)
        self.assertEqual(len(spec), 200)


class TestIO(unittest.TestCase):
    def test_roundtrip_utf8_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.csv"
            save_csv(path, {"raman_shift_cm-1": np.array([400.0, 401.0]),
                            "intensity": np.array([10.0, 11.0])})
            spec = load_csv(path)
            self.assertEqual(len(spec), 2)
            np.testing.assert_allclose(spec.y, [10.0, 11.0])

    def test_reads_gbk_encoded_with_chinese_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gbk.csv"
            path.write_bytes("拉曼位移,强度\n400,10\n401,11\n".encode("gbk"))
            spec = load_csv(path)
            self.assertEqual(len(spec), 2)

    def test_skips_blank_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("shift,intensity\n400,1\n\n401,2\n402,3\n",
                            encoding="utf-8")
            spec = load_csv(path)
            self.assertEqual(len(spec), 3)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("definitely_missing_file.csv")


class TestPreprocess(unittest.TestCase):
    def test_remove_baseline_shape_and_reduction(self):
        spec = toy_spectrum()
        corrected = remove_baseline(spec, "imodpoly", poly_order=3)
        self.assertEqual(len(corrected), len(spec))
        self.assertIn("baseline", corrected.meta)
        # 校正后远离峰的区域应接近 0
        flat = corrected.y[(corrected.x > 1250) & (corrected.x < 1550)]
        self.assertLess(float(np.mean(np.abs(flat))), 30.0)

    def test_unknown_method_raises(self):
        spec = toy_spectrum()
        with self.assertRaises(ValueError):
            remove_baseline(spec, "not_a_method")


class TestAugment(unittest.TestCase):
    def test_shapes_and_labels_preserved(self):
        X = np.random.default_rng(0).normal(size=(20, 50))
        y = np.array(["a", "b"] * 10)
        g = np.array(["b1"] * 10 + ["b2"] * 10)
        Xa, ya, ga = augment(X, y, g, n_copies=3, seed=0)
        self.assertEqual(Xa.shape, (60, 50))
        self.assertEqual(ya.shape, (60,))
        self.assertEqual(ga.shape, (60,))

    def test_no_copies_returns_original(self):
        X = np.zeros((4, 5))
        y = np.array(["a"] * 4)
        g = np.array(["b"] * 4)
        Xa, _, _ = augment(X, y, g, n_copies=1)
        np.testing.assert_allclose(Xa, X)


class TestPrototype(unittest.TestCase):
    def make_blobs(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (30, 4)),
                       rng.normal(5, 0.3, (30, 4))])
        y = np.array(["a"] * 30 + ["b"] * 30)
        return X, y

    def test_separable_blobs_perfect(self):
        X, y = self.make_blobs()
        clf = NearestPrototype().fit(X, y)
        self.assertEqual(float(np.mean(clf.predict(X) == y)), 1.0)

    def test_cosine_variant_runs(self):
        X, y = self.make_blobs()
        clf = NearestPrototype(metric="cosine").fit(X, y)
        # 余弦只看方向、忽略幅值；这里两个簇位于同一方向附近，因此弱于欧氏属正常
        self.assertGreater(float(np.mean(clf.predict(X) == y)), 0.6)


class TestOpenSet(unittest.TestCase):
    def make_data(self):
        rng = np.random.default_rng(1)
        known = np.vstack([rng.normal(0, 0.3, (60, 3)),
                           rng.normal(6, 0.3, (60, 3))])
        y = np.array(["a"] * 60 + ["b"] * 60)
        far = rng.normal(60, 0.3, (30, 3))        # 远离已知类的未知物
        near = rng.normal(3, 0.3, (30, 3))        # 落在两类之间的未知物
        return known, y, far, near

    def test_score_keys_and_length(self):
        X, y, far, _ = self.make_data()
        model = OpenSetModel().fit(X, y)
        scores = model.score(far)
        self.assertEqual(set(scores), {"margin", "prototype", "mahalanobis"})
        self.assertEqual(scores["mahalanobis"].shape, (30,))

    def test_far_unknown_is_easy(self):
        from sklearn.metrics import roc_auc_score
        X, y, far, _ = self.make_data()
        model = OpenSetModel().fit(X, y)
        s_known = model.score(X)["mahalanobis"]
        s_far = model.score(far)["mahalanobis"]
        auc_far = roc_auc_score(np.r_[np.ones(len(s_known)), np.zeros(len(s_far))],
                                np.r_[s_known, s_far])
        self.assertGreater(auc_far, 0.95)

    def test_too_typical_unknown_is_undetectable(self):
        """复现 P7 的核心发现：比已知样本更"典型"的未知物无法被拒识。

        这里把未知样本直接取成某一类的中心点——它到类中心的距离为 0，
        比任何真实已知样本都更靠近，因此得分更高、AUROC 低于 0.5（反向）。
        实测数据里的 S-warneri（AUROC 0.10）就是这个情形。
        """
        from sklearn.metrics import roc_auc_score
        X, y, _, _ = self.make_data()
        model = OpenSetModel().fit(X, y)
        centroid = X[y == "a"].mean(axis=0, keepdims=True)
        too_typical = np.repeat(centroid, 20, axis=0)

        s_known = model.score(X)["mahalanobis"]
        s_tt = model.score(too_typical)["mahalanobis"]
        auc = roc_auc_score(np.r_[np.ones(len(s_known)), np.zeros(len(s_tt))],
                            np.r_[s_known, s_tt])
        self.assertLess(auc, 0.5)
        self.assertGreater(float(np.mean(s_tt)), float(np.mean(s_known)))

    def test_threshold_quantile(self):
        scores = np.arange(100, dtype=float)
        tau = threshold_at(scores, keep_rate=0.95)
        self.assertAlmostEqual(tau, np.quantile(scores, 0.05))
        self.assertGreater(tau, scores.min())
        self.assertLess(tau, scores.max())


if __name__ == "__main__":
    unittest.main(verbosity=2)
