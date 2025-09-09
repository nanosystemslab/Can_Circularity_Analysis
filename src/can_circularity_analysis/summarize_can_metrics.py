#!/usr/bin/env python3
# pip install pandas numpy matplotlib
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ======================= I/O + flattening =======================


def read_metric_file(fp: Path) -> dict:
    """Flatten one *_metrics.json into a single-row dict; tolerant to missing fields."""
    try:
        with open(fp) as f:
            d = json.load(f)
    except Exception as e:
        return {"_read_error": str(e), "_file": str(fp)}

    row = {"_file": str(fp)}
    # core flat fields
    for k, v in d.items():
        if isinstance(v, str | int | float | bool) or v is None:
            row[k] = v

    # flatten centers
    if isinstance(d.get("center_px"), list | tuple) and len(d["center_px"]) == 2:
        row["center_px_x"], row["center_px_y"] = d["center_px"]
    if isinstance(d.get("center_mm"), list | tuple) and len(d["center_mm"]) == 2:
        row["center_mm_x"], row["center_mm_y"] = d["center_mm"]

    # ellipse flatten
    ell = d.get("ellipse")
    if isinstance(ell, dict):
        for k, v in ell.items():
            if isinstance(v, str | int | float | bool) or v is None:
                row[f"ellipse_{k}"] = v
            elif (
                isinstance(v, list | tuple)
                and len(v) == 2
                and all(isinstance(x, int | float) for x in v)
            ):
                row[f"ellipse_{k}_x"] = v[0]
                row[f"ellipse_{k}_y"] = v[1]

    # derive helpers
    img = d.get("image")
    if img:
        p = Path(img)
        row["_image_name"] = p.name
        row["_parent_dir"] = p.parent.name
        row["_stem"] = p.stem

    # quick success flag
    row["_ok"] = (row.get("overlay_image") is not None) and (row.get("_read_error") is None)
    return row


def summarize_numeric(df: pd.DataFrame, groupby=None) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    if groupby is None:
        return numeric.describe().T
    g = df.groupby(groupby)
    parts = []
    for name, sub in g:
        desc = sub.select_dtypes(include=[np.number]).describe().T
        desc.insert(0, groupby, name)
        parts.append(desc)
    out = pd.concat(parts, axis=0)
    out.reset_index(names=["metric"], inplace=True)
    return out


# ======================= plotting helpers =======================


def load_case_for_plots(metrics_fp: Path):
    """Load metrics + its points CSV for plotting overlays and polar Δr(θ)."""
    with open(metrics_fp) as f:
        m = json.load(f)
    csv_fp = Path(m["csv_points"])
    df = pd.read_csv(csv_fp)

    have_mm = ("pixels_per_mm" in m) and m["pixels_per_mm"]
    radius = float(m["radius_px"])
    cx, cy = float(m["center_px"][0]), float(m["center_px"][1])

    X = df["x_px"].to_numpy() - cx
    Y = df["y_px"].to_numpy() - cy
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    dev_px = R - radius

    mm_per_px = 1.0 / float(m["pixels_per_mm"]) if have_mm else None
    dev_mm = dev_px * mm_per_px if have_mm else None

    ell = m.get("ellipse")
    ellipse = None
    if isinstance(ell, dict) and ("major_px" in ell) and ("minor_px" in ell):
        ellipse = {
            "MA_px": float(ell["major_px"]),
            "ma_px": float(ell["minor_px"]),
            "angle_deg": float(ell["angle_deg"]),
        }
        if have_mm:
            ellipse["MA_mm"] = ell.get("major_mm", ellipse["MA_px"] * mm_per_px)
            ellipse["ma_mm"] = ell.get("minor_mm", ellipse["ma_px"] * mm_per_px)

    return {
        "stem": Path(m["image"]).stem,
        "parent": Path(m["image"]).parent.name if "image" in m else "",
        "X": X,
        "Y": Y,
        "R": R,
        "theta": theta,
        "radius_px": radius,
        "dev_px": dev_px,
        "dev_mm": dev_mm,
        "has_mm": have_mm,
        "ellipse": ellipse,
        "diameter_mm": m.get("diameter_mm", None),
        "rms_mm": m.get("rms_out_of_round_mm", None),
        "std_mm": m.get("std_out_of_round_mm", None),
        "range_mm": m.get("range_out_of_round_mm", None),
    }


def unit_normalize_points(X, Y, radius_px):
    s = 1.0 / radius_px
    return X * s, Y * s


def draw_unit_circle(ax):
    t = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), alpha=0.4)


def overlay_circles(cases, out_png, legend=False, legend_limit=10, legend_by_parent=False):
    plt.figure()
    ax = plt.gca()
    theta = np.linspace(0, 2 * np.pi, 400)
    base_x = np.cos(theta)
    base_y = np.sin(theta)
    for i, c in enumerate(cases):
        label = None
        if legend and i < legend_limit:
            label = c["parent"] if legend_by_parent else c["stem"]
        # all normalized circles look the same; we still plot one per case for alpha stacking
        ax.plot(base_x, base_y, alpha=0.3, label=label)
    ax.set_aspect("equal", "box")
    ax.set_title("Overlay: Best-fit circles (normalized)")
    ax.set_xlabel("x / r_fit")
    ax.set_ylabel("y / r_fit")
    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="x-small",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()


def overlay_rims(
    cases, out_png, max_points_per=400, legend=False, legend_limit=10, legend_by_parent=False
):
    plt.figure()
    ax = plt.gca()
    for i, c in enumerate(cases):
        Xn, Yn = unit_normalize_points(c["X"], c["Y"], c["radius_px"])
        if len(Xn) > max_points_per:
            idx = np.linspace(0, len(Xn) - 1, max_points_per).astype(int)
        else:
            idx = np.arange(len(Xn))
        label = None
        if legend and i < legend_limit:
            label = c["parent"] if legend_by_parent else c["stem"]
        ax.plot(Xn[idx], Yn[idx], ".", markersize=1, label=label)
    draw_unit_circle(ax)
    ax.set_aspect("equal", "box")
    ax.set_title("Overlay: Rim points (normalized to unit circle)")
    ax.set_xlabel("x / r_fit")
    ax.set_ylabel("y / r_fit")
    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="x-small",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()


def overlay_ellipses(cases, out_png, legend=False, legend_limit=10, legend_by_parent=False):
    def ellipse_pts(MA, ma, ang_deg, n=400):
        a = MA / 2.0
        b = ma / 2.0
        t = np.linspace(0, 2 * np.pi, n)
        x = a * np.cos(t)
        y = b * np.sin(t)
        th = np.deg2rad(ang_deg)
        xr = x * np.cos(th) - y * np.sin(th)
        yr = x * np.sin(th) + y * np.cos(th)
        return xr, yr

    plt.figure()
    ax = plt.gca()
    drew_any = False
    for i, c in enumerate(cases):
        e = c["ellipse"]
        if not e:
            continue
        xr, yr = ellipse_pts(
            e["MA_px"] / c["radius_px"], e["ma_px"] / c["radius_px"], e["angle_deg"]
        )
        label = None
        if legend and i < legend_limit:
            label = c["parent"] if legend_by_parent else c["stem"]
        ax.plot(xr, yr, label=label)
        drew_any = True
    draw_unit_circle(ax)
    ax.set_aspect("equal", "box")
    ax.set_title("Overlay: Best-fit ellipses (normalized)")
    ax.set_xlabel("x / r_fit")
    ax.set_ylabel("y / r_fit")
    if drew_any and legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="x-small",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()


def polar_deviation(
    cases,
    out_png,
    use_mm=False,
    show_mean_band=True,
    legend=False,
    legend_limit=10,
    legend_by_parent=False,
):
    plt.figure()
    ax = plt.gca()
    all_theta, all_dev = [], []
    for i, c in enumerate(cases):
        th = c["theta"]
        dev = c["dev_mm"] if (use_mm and c["has_mm"]) else c["dev_px"]
        label = None
        if legend and i < legend_limit:
            label = c["parent"] if legend_by_parent else c["stem"]
        ax.plot(np.degrees(th), dev, label=label)
        all_theta.append(th)
        all_dev.append(dev)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Δr ({})".format("mm" if use_mm else "px"))
    ax.set_title("Δr(θ) for all rims")

    if show_mean_band:
        th_grid = np.linspace(-180, 180, 361)
        curves = []
        for th, dv in zip(all_theta, all_dev, strict=False):
            thd = np.degrees(th)
            order = np.argsort(thd)
            thd = thd[order]
            dvd = dv[order]
            curves.append(np.interp(th_grid, thd, dvd, left=np.nan, right=np.nan))
        arr = np.array(curves)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ax.plot(th_grid, mean, linewidth=2)
        ax.fill_between(
            th_grid, mean - std, mean + std, alpha=0.2, label="mean ± std" if legend else None
        )

    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="x-small",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()


def histograms(cases, out_dir):
    diam_mm = [c["diameter_mm"] for c in cases if c["diameter_mm"] is not None]
    rms_mm = [c["rms_mm"] for c in cases if c["rms_mm"] is not None]
    std_mm = [c["std_mm"] for c in cases if c["std_mm"] is not None]
    rng_mm = [c["range_mm"] for c in cases if c["range_mm"] is not None]

    if len(diam_mm) > 0:
        plt.figure()
        plt.hist(diam_mm, bins=20)
        plt.xlabel("Diameter (mm)")
        plt.ylabel("Count")
        plt.title("Diameter distribution")
        plt.savefig(out_dir / "hist_diameter_mm.png", bbox_inches="tight", dpi=150)
        plt.close()

    if len(rms_mm) > 0:
        plt.figure()
        plt.hist(rms_mm, bins=20)
        plt.xlabel("RMS out-of-round (mm)")
        plt.ylabel("Count")
        plt.title("RMS out-of-round")
        plt.savefig(out_dir / "hist_rms_mm.png", bbox_inches="tight", dpi=150)
        plt.close()

    if len(std_mm) > 0:
        plt.figure()
        plt.hist(std_mm, bins=20)
        plt.xlabel("STD out-of-round (mm)")
        plt.ylabel("Count")
        plt.title("STD out-of-round")
        plt.savefig(out_dir / "hist_std_mm.png", bbox_inches="tight", dpi=150)
        plt.close()

    if len(rng_mm) > 0:
        plt.figure()
        plt.hist(rng_mm, bins=20)
        plt.xlabel("RANGE out-of-round (mm)")
        plt.ylabel("Count")
        plt.title("RANGE out-of-round")
        plt.savefig(out_dir / "hist_range_mm.png", bbox_inches="tight", dpi=150)
        plt.close()


def scatter_ellipse_axes(cases, out_png, legend=False, legend_limit=10, legend_by_parent=False):
    xs, ys, labels = [], [], []
    for c in cases:
        e = c["ellipse"]
        if not e:
            continue
        if ("MA_mm" in e) and ("ma_mm" in e):
            xs.append(e["ma_mm"])
            ys.append(e["MA_mm"])
            labels.append(c["parent"] if legend_by_parent else c["stem"])
        else:
            if ("MA_px" in e) and ("ma_px" in e):
                xs.append(e["ma_px"])
                ys.append(e["MA_px"])
                labels.append(c["parent"] if legend_by_parent else c["stem"])
    if len(xs) == 0:
        plt.figure()
        plt.text(0.5, 0.5, "No ellipse data", ha="center")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close()
        return
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    plt.figure()
    plt.plot(xs, ys, ".", markersize=4, label=None)
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Minor axis")
    plt.ylabel("Major axis")
    plt.title("Ellipse axes (mm if available)")
    if legend and len(labels) > 0:
        for i in range(min(len(xs), legend_limit)):
            plt.annotate(labels[i], (xs[i], ys[i]), fontsize=6, alpha=0.8)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()


# ======================= main =======================


def main():
    ap = argparse.ArgumentParser(
        description="Summarize can metrics (CSV stats) and generate overlay plots."
    )
    ap.add_argument("--in-dir", default="out", help="Directory containing *_metrics.json")
    ap.add_argument("--pattern", default="*_metrics.json", help="Glob pattern for metrics")
    ap.add_argument("--out-dir", default="out", help="Where to write CSV summaries")
    ap.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="If >0, also group stats by filename prefix of this length.",
    )
    # plotting
    ap.add_argument(
        "--make-plots", action="store_true", help="Also generate plots into <out-dir>/plots"
    )
    ap.add_argument(
        "--polar-mm", action="store_true", help="Polar Δr(θ) in mm (requires scale in JSONs)"
    )
    ap.add_argument(
        "--max-points-per", type=int, default=400, help="Max points per rim in overlays"
    )
    # legends
    ap.add_argument(
        "--legend", action="store_true", help="Add legends to overlay plots (may clutter)."
    )
    ap.add_argument(
        "--legend-limit", type=int, default=10, help="Max labels/entries to show in legends."
    )
    ap.add_argument(
        "--legend-by-parent",
        action="store_true",
        help="Legend labels use parent directory name instead of filename stem.",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f("[warn] No files matched {in_dir}/{args.pattern}"), file=sys.stderr)

    # --------- flatten to table ----------
    rows = [read_metric_file(fp) for fp in files]
    df = pd.DataFrame(rows)

    # attempt numeric coercion for convenient stats
    for col in list(df.columns):
        if not col.startswith("_"):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # helpful derived
    if "diameter_mm" not in df.columns and "radius_mm" in df.columns:
        df["diameter_mm"] = df["radius_mm"] * 2.0

    # write master table
    all_csv = out_dir / "metrics_all.csv"
    df.to_csv(all_csv, index=False)
    print(f"[ok] wrote {all_csv}")

    # overall stats
    stats_overall = summarize_numeric(df)
    stats_overall_csv = out_dir / "stats_overall.csv"
    stats_overall.to_csv(stats_overall_csv)
    print(f"[ok] wrote {stats_overall_csv}")

    # by parent dir
    if "_parent_dir" in df.columns:
        stats_by_parent = summarize_numeric(df, groupby="_parent_dir")
        stats_by_parent_csv = out_dir / "stats_by_parent.csv"
        stats_by_parent.to_csv(stats_by_parent_csv, index=False)
        print(f"[ok] wrote {stats_by_parent_csv}")

    # optional: by filename prefix
    if args.prefix_len and "_stem" in df.columns:
        df["_prefixN"] = df["_stem"].str.slice(0, args.prefix_len)
        stats_by_prefix = summarize_numeric(df, groupby="_prefixN")
        stats_by_prefix_csv = out_dir / "stats_by_prefix.csv"
        stats_by_prefix.to_csv(stats_by_prefix_csv, index=False)
        print(f"[ok] wrote {stats_by_prefix_csv}")

    # --------- plots ----------
    if args.make_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # load cases (metrics + points)
        cases = []
        for fp in files:
            try:
                cases.append(load_case_for_plots(fp))
            except Exception as e:
                print(f"[skip plotting] {fp}: {e}", file=sys.stderr)
        if not cases:
            print("[warn] no cases to plot")
            return

        # NEW: overlay of best-fit circles (normalized)
        overlay_circles(
            cases,
            plots_dir / "overlay_circles.png",
            legend=args.legend,
            legend_limit=args.legend_limit,
            legend_by_parent=args.legend_by_parent,
        )

        # overlays (rims + ellipses)
        overlay_rims(
            cases,
            plots_dir / "overlay_rims.png",
            max_points_per=args.max_points_per,
            legend=args.legend,
            legend_limit=args.legend_limit,
            legend_by_parent=args.legend_by_parent,
        )
        overlay_ellipses(
            cases,
            plots_dir / "overlay_ellipses.png",
            legend=args.legend,
            legend_limit=args.legend_limit,
            legend_by_parent=args.legend_by_parent,
        )

        # polar Δr(θ)
        polar_deviation(
            cases,
            plots_dir / ("polar_deviation_mm.png" if args.polar_mm else "polar_deviation_px.png"),
            use_mm=args.polar_mm,
            show_mean_band=True,
            legend=args.legend,
            legend_limit=args.legend_limit,
            legend_by_parent=args.legend_by_parent,
        )

        # histograms + scatter
        histograms(cases, plots_dir)
        scatter_ellipse_axes(
            cases,
            plots_dir / "scatter_ellipse_axes.png",
            legend=args.legend,
            legend_limit=args.legend_limit,
            legend_by_parent=args.legend_by_parent,
        )

        print(f"[ok] Plots written to {plots_dir}")

    # quick run summary
    tried = len(df)
    ok = int(df["_ok"].sum()) if "_ok" in df.columns else tried
    fail = tried - ok
    print(f"[summary] files: {tried}, ok: {ok}, failed: {fail}")


if __name__ == "__main__":
    main()
