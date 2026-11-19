"""raman-rec 命令行入口。

面向"使用"的统一入口；`05_*.py`～`10_*.py` 是面向"实验复现"的脚本，两者分工不同：
命令行做日常处理，脚本保证研究结果可复现。

用法示例
    # 查看数据集概览
    python cli.py info --dataset bacterial

    # 校正一条自己的光谱（两列 CSV）
    python cli.py preprocess my_raman.csv --method imodpoly --order 5

    # 跑一次分类基线（两种协议都报）
    python cli.py classify --dataset oil
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------- 子命令：info
def cmd_info(args) -> int:
    from raman_rec import load_bacterial, load_oil

    loaders = {"bacterial": load_bacterial, "oil": load_oil}
    path = {"bacterial": DATA_DIR / "Bacterial_data",
            "oil": DATA_DIR / "Oil_data"}
    if args.dataset == "both":
        names = ["bacterial", "oil"]
    else:
        names = [args.dataset]

    for name in names:
        ds = loaders[name](path[name])
        print("=" * 68)
        print(ds.summary())
    return 0


# ---------------------------------------------------------------- 子命令：preprocess
def cmd_preprocess(args) -> int:
    import matplotlib
    if args.no_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from raman_rec import load_csv, remove_baseline, save_csv

    spec = load_csv(args.csv)
    corrected = remove_baseline(spec, args.method, **(
        {"poly_order": args.order} if args.method in ("modpoly", "imodpoly") else {}))
    if args.normalize != "none":
        corrected = corrected.normalized(args.normalize)

    out_csv = Path(args.out) if args.out else Path(args.csv).with_name(
        Path(args.csv).stem + "_corrected.csv")
    save_csv(out_csv, {
        "raman_shift_cm-1": spec.x,
        "raw_intensity": spec.y,
        "baseline": corrected.meta["baseline"],
        "corrected_intensity": corrected.y,
    })

    print(f"输入      : {args.csv}")
    print(f"点数      : {len(spec)}（{spec.x[0]:.1f}~{spec.x[-1]:.1f} cm-1）")
    print(f"方法      : {args.method}" + (f"（poly_order={args.order}）"
                                        if args.method in ("modpoly", "imodpoly") else ""))
    print(f"迭代次数  : {corrected.meta.get('baseline_iterations', '-')}")
    print(f"负值比例  : {corrected.negative_ratio():.4f}")
    print(f"SNR       : {spec.snr():.1f}")
    peaks = corrected.peaks(prominence_frac=0.1)
    print(f"检出峰数  : {len(peaks['index'])}"
          + (f"（前三个峰位：{', '.join(f'{v:.0f}' for v in peaks['shift'][:3])}）"
             if len(peaks["index"]) else ""))
    print(f"输出 CSV  : {out_csv}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.plot(spec.x, spec.y, color="0.6", lw=1, label="原始谱")
        ax.plot(spec.x, corrected.meta["baseline"], color="#E45756", lw=1.6,
                label=f"{args.method} 基线")
        ax.plot(spec.x, corrected.y, color="#4C78A8", lw=1.1, label="校正后")
        ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
        ax.set_ylabel("强度")
        ax.legend()
        fig.tight_layout()
        png = out_csv.with_suffix(".png")
        fig.savefig(png, dpi=170)
        print(f"输出图    : {png}")
        plt.show()
    return 0


# ---------------------------------------------------------------- 子命令：classify
def cmd_classify(args) -> int:
    from raman_rec import load_bacterial, load_oil
    from raman_rec.classify import group_cv_eval, random_split_eval

    ds = (load_bacterial(DATA_DIR / "Bacterial_data") if args.dataset == "bacterial"
          else load_oil(DATA_DIR / "Oil_data"))
    print(ds.summary())
    print("-" * 68)

    rand = random_split_eval(ds, test_size=0.3, seed=0,
                             n_components=args.components,
                             classifier=args.classifier)
    print(f"随机分层划分 : 准确率 {rand['accuracy']:.3f} | 宏平均 F1 {rand['macro_f1']:.3f}")

    if any(s.meta.get("batch") for s in ds.spectra):
        grouped = group_cv_eval(ds, group_key="batch",
                                n_components=args.components,
                                classifier=args.classifier)
        print(f"留一批次验证 : 平均 {grouped['mean_accuracy']:.3f} "
              f"± {grouped['std_accuracy']:.3f}（{len(grouped['folds'])} 折）")
        print(f"两者差距     : {rand['accuracy'] - grouped['mean_accuracy']:+.3f}")
    return 0


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py", description="拉曼光谱处理与识别工具（raman-rec）")
    p.add_argument("--version", action="version", version="raman-rec 0.3.0")
    sub = p.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="打印数据集概览")
    p_info.add_argument("--dataset", choices=["bacterial", "oil", "both"],
                        default="both")
    p_info.set_defaults(func=cmd_info)

    p_pre = sub.add_parser("preprocess", help="校正一条两列 CSV 光谱")
    p_pre.add_argument("csv", help="输入 CSV（拉曼位移, 强度）")
    p_pre.add_argument("--method", default="imodpoly",
                       choices=["modpoly", "imodpoly", "asls", "airpls",
                                "snip", "mixture_model", "pspline_asls"])
    p_pre.add_argument("--order", type=int, default=5,
                       help="多项式阶数，仅 modpoly/imodpoly 有效")
    p_pre.add_argument("--normalize", default="none",
                       choices=["none", "snv", "max", "minmax", "area", "l2"])
    p_pre.add_argument("--out", default=None, help="输出 CSV 路径")
    p_pre.add_argument("--no-plot", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_cls = sub.add_parser("classify", help="跑一次分类基线（两种协议）")
    p_cls.add_argument("--dataset", choices=["bacterial", "oil"], default="bacterial")
    p_cls.add_argument("--components", type=int, default=None,
                       help="PCA 维数，省略则不降维（推荐）")
    p_cls.add_argument("--classifier", choices=["svc", "lda", "logreg"],
                       default="svc")
    p_cls.set_defaults(func=cmd_classify)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
