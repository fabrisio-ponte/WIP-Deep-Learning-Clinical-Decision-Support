#!/usr/bin/env python3
"""Generate publication figures from fixed manuscript values.

This keeps the visualizations reproducible and synchronized with main.tex.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "behrt": "#D55E00",
    "other": "#4C78A8",
    "positive": "#2E8B57",
    "negative": "#C44E52",
    "neutral": "#7A7A7A",
    "recurring": "#4C78A8",
    "new": "#F28E2B",
}


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)


def save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_posweight_tradeoff() -> None:
    seeds = np.array([42, 43, 44])
    d_aps = np.array([0.0193, 0.0144, 0.0269])
    d_auc = np.array([0.0096, 0.0105, 0.0120])
    d_f1 = np.array([-0.0335, -0.0343, -0.0223])

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharex=True)
    panels = [
        (d_aps, "Delta APS", COLORS["positive"]),
        (d_auc, "Delta AUC", COLORS["positive"]),
        (d_f1, "Delta micro-F1", COLORS["negative"]),
    ]

    for ax, (vals, title, color) in zip(axes, panels):
        ax.axhline(0.0, color=COLORS["neutral"], lw=1.0, alpha=0.9)
        ax.plot(seeds, vals, marker="o", color=color, lw=2.2)
        ax.scatter(seeds, vals, color=color, s=42, zorder=3)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel("Seed")
        style_axis(ax)

        # Give labels enough vertical room even when per-seed deltas are close.
        ymin, ymax = float(vals.min()), float(vals.max())
        yrange = ymax - ymin
        ylim_pad = max(yrange * 0.45, 0.0018)
        ax.set_ylim(ymin - ylim_pad, ymax + ylim_pad)

        y_offset_pts = 8 if vals.mean() >= 0 else -14
        va = "bottom" if vals.mean() >= 0 else "top"
        for x, y in zip(seeds, vals):
            ax.annotate(
                f"{y:+.4f}",
                xy=(x, y),
                xytext=(0, y_offset_pts),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.9},
                zorder=4,
            )

    axes[0].set_ylabel("Weighted - baseline")
    fig.suptitle("Positive-class weighting effects across seeds", fontsize=11, y=1.03)
    fig.tight_layout()
    save(fig, "fig_posweight_tradeoff.png")


def fig_bootstrap_ci() -> None:
    models = [
        "XGBoost",
        "LogReg",
        "BEHRT",
        "Popularity",
        "Repeat-last",
    ]

    aps_mean = np.array([0.4894, 0.2915, 0.2735, 0.2544, 0.2372])
    aps_lo = np.array([0.4863, 0.2890, 0.2708, 0.2521, 0.2346])
    aps_hi = np.array([0.4924, 0.2939, 0.2762, 0.2569, 0.2398])

    auc_mean = np.array([0.9307, 0.9162, 0.9006, 0.8914, 0.7184])
    auc_lo = np.array([0.9297, 0.9153, 0.8997, 0.8902, 0.7165])
    auc_hi = np.array([0.9318, 0.9171, 0.9016, 0.8924, 0.7202])

    f1_mean = np.array([0.1467, 0.2966, 0.2381, 0.2716, 0.4264])
    f1_lo = np.array([0.1455, 0.2946, 0.2358, 0.2691, 0.4237])
    f1_hi = np.array([0.1479, 0.2986, 0.2403, 0.2740, 0.4292])

    metrics = [
        (aps_mean, aps_lo, aps_hi, "APS"),
        (auc_mean, auc_lo, auc_hi, "ROC-AUC"),
        (f1_mean, f1_lo, f1_hi, "Micro-F1"),
    ]

    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)

    for ax, (mean, lo, hi, title) in zip(axes, metrics):
        err = np.vstack([mean - lo, hi - mean])
        for i, m in enumerate(models):
            color = COLORS["behrt"] if m == "BEHRT" else COLORS["other"]
            marker = "D" if m == "BEHRT" else "o"
            ax.errorbar(
                mean[i],
                y[i],
                xerr=err[:, i : i + 1],
                fmt=marker,
                color=color,
                elinewidth=1.4,
                capsize=3,
                markersize=5.8,
            )
        ax.set_title(title, fontsize=10, pad=8)
        style_axis(ax)
        ax.grid(axis="y", visible=False)
        margin = (mean.max() - mean.min()) * 0.22
        ax.set_xlim(mean.min() - margin, mean.max() + margin)
        for i, m in enumerate(models):
            if m == "BEHRT":
                ax.text(mean[i], y[i] - 0.3, "BEHRT", ha="center", va="bottom", fontsize=8, color=COLORS["behrt"])

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    fig.suptitle("Bootstrap 95% confidence intervals (500 paired resamples)", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, "fig_bootstrap_ci.png")


def fig_new_vs_recurring() -> None:
    models = ["XGBoost", "LogReg", "BEHRT"]

    recur_f1 = np.array([0.440, 0.453, 0.403])
    new_f1 = np.array([0.050, 0.025, 0.061])
    recur_auc = np.array([0.760, 0.676, 0.681])
    new_auc = np.array([0.855, 0.838, 0.865])

    x = np.arange(len(models))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True)

    b1 = axes[0].bar(x - w / 2, recur_f1, width=w, label="Recurring", color=COLORS["recurring"])
    b2 = axes[0].bar(x + w / 2, new_f1, width=w, label="New", color=COLORS["new"])
    axes[0].set_title("F1 by target type", fontsize=10, pad=8)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 0.5)
    style_axis(axes[0])
    axes[0].grid(axis="x", visible=False)

    b3 = axes[1].bar(x - w / 2, recur_auc, width=w, label="Recurring", color=COLORS["recurring"])
    b4 = axes[1].bar(x + w / 2, new_auc, width=w, label="New", color=COLORS["new"])
    axes[1].set_title("ROC-AUC by target type", fontsize=10, pad=8)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.6, 0.9)
    style_axis(axes[1])
    axes[1].grid(axis="x", visible=False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models)

    fig.legend(
        [b1[0], b2[0]],
        ["Recurring", "New"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.5,
    )

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax = bar.axes
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.008 if h < 0.5 else 0.004),
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
            )

    fig.suptitle("Model behavior differs sharply on recurring vs new diagnoses", fontsize=11, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_new_vs_recurring.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.facecolor": "#F8F8F8",
            "figure.facecolor": "white",
        }
    )

    fig_posweight_tradeoff()
    fig_bootstrap_ci()
    fig_new_vs_recurring()


if __name__ == "__main__":
    main()
