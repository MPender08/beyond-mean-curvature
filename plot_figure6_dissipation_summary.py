from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

THERMO_DIR = PROJECT_ROOT / "results" / "selective_tail" / "aggregated" / "thermo_stress_test"

INPUT_CSV = THERMO_DIR / "thermo_ranking_table.csv"

MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "results" / "selective_tail" / "manuscript_figures"

OUT_DIR = MANUSCRIPT_FIG_DIR
OUT_FILE = OUT_DIR / "figure6_dissipation_summary.png"

MODEL_ORDER = [
    "sum_load",
    "sum_load_1p5",
    "sum_load_sq",
    "sum_load_cu",
]

MODEL_LABELS = {
    "sum_load": r"$\sum load$",
    "sum_load_1p5": r"$\sum load^{1.5}$",
    "sum_load_sq": r"$\sum load^2$",
    "sum_load_cu": r"$\sum load^3$",
}

N_ACTIVE_ORDER = [5, 10, 20]


# ============================================================
# Helpers
# ============================================================
def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ranking_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "dissipation_model",
        "row_type",
        "placement",
        "n_active",
        "value",
        "note",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["dissipation_model"].isin(MODEL_ORDER)].copy()
    if df.empty:
        raise ValueError("No rows found for the requested dissipation models.")

    df["dissipation_model"] = pd.Categorical(
        df["dissipation_model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    df["n_active"] = df["n_active"].astype(int)

    return df.sort_values(["dissipation_model", "row_type", "placement", "n_active"]).reset_index(drop=True)


def prepare_optimum_panel(df: pd.DataFrame) -> pd.DataFrame:
    opt_df = df[df["row_type"] == "placement_optimum"].copy()
    if opt_df.empty:
        raise ValueError("No placement_optimum rows found.")
    return opt_df.sort_values(["dissipation_model", "placement"]).reset_index(drop=True)


def prepare_gap_panel(df: pd.DataFrame) -> pd.DataFrame:
    gap_df = df[df["row_type"] == "matched_gap"].copy()
    if gap_df.empty:
        raise ValueError("No matched_gap rows found.")
    return gap_df.sort_values(["dissipation_model", "n_active"]).reset_index(drop=True)


def draw_gap_bars(ax: plt.Axes, gap_df: pd.DataFrame, x: np.ndarray, width: float, offsets: dict[int, float]) -> list[float]:
    y_values: list[float] = []

    for n in N_ACTIVE_ORDER:
        sub = (
            gap_df[gap_df["n_active"] == n]
            .set_index("dissipation_model")
            .reindex(MODEL_ORDER)
        )
        vals = sub["value"].to_numpy(dtype=float)
        y_values.extend(vals.tolist())

        ax.bar(
            x + offsets[n],
            vals,
            width=width,
            label=f"n={n}",
        )

    return y_values


# ============================================================
# Plotting
# ============================================================
def plot_summary_figure(df: pd.DataFrame) -> Path:
    ensure_output_dir()

    opt_df = prepare_optimum_panel(df)
    gap_df = prepare_gap_panel(df)

    x = np.arange(len(MODEL_ORDER))

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.55, 0.32], hspace=0.34)

    # --------------------------------------------------------
    # Panel A
    # --------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    targeted = (
        opt_df[opt_df["placement"] == "targeted"]
        .set_index("dissipation_model")
        .reindex(MODEL_ORDER)
    )
    random = (
        opt_df[opt_df["placement"] == "random"]
        .set_index("dissipation_model")
        .reindex(MODEL_ORDER)
    )

    x_targeted = x - 0.04
    x_random = x + 0.04

    ax1.plot(
        x_targeted,
        targeted["n_active"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Targeted optimum",
    )
    ax1.plot(
        x_random,
        random["n_active"].to_numpy(dtype=float),
        marker="s",
        linewidth=2,
        label="Random optimum",
    )

    ax1.set_ylabel("Best n_active")
    ax1.set_yticks(N_ACTIVE_ORDER)
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.set_title("A. Preferred shortcut regime shifts with hotspot sensitivity", loc="left", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="upper left")

    # --------------------------------------------------------
    # Panel B
    # --------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    width = 0.22
    offsets = {
        5: -width,
        10: 0.0,
        20: width,
    }

    y_values = draw_gap_bars(ax2, gap_df, x, width, offsets)

    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_ylabel(r"$\Delta J_{\mathrm{thermo}}$" + "\n(targeted - random)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([])
    ax2.set_title("B. Ranking becomes regime-dependent under superlinear dissipation", loc="left", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)

    max_abs = max(abs(v) for v in y_values if pd.notna(v))
    linthresh = max(1.0, 0.05 * max_abs)
    ax2.set_yscale("symlog", linthresh=linthresh)

    ax2.text(
        0.985,
        0.96,
        "negative = targeted better\npositive = random better",
        transform=ax2.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.01))

    # Inset zoom: first three models only, linear scale for readability.
    axins = inset_axes(
        ax2,
        width="40%",
        height="42%",
        loc="lower left",
        bbox_to_anchor=(0.10, 0.08, 0.9, 0.9),
        bbox_transform=ax2.transAxes,
        borderpad=0.8,
    )

    draw_gap_bars(axins, gap_df, x, width, offsets)
    axins.axhline(0, linestyle="--", linewidth=1)
    axins.grid(True, axis="y", alpha=0.25)

    # Zoom to the first three models and use a data-driven linear y-range.
    zoom_vals = gap_df[gap_df["dissipation_model"].isin(MODEL_ORDER[:3])]["value"].to_numpy(dtype=float)
    zoom_max = np.nanmax(np.abs(zoom_vals))
    zoom_pad = max(5.0, 0.25 * zoom_max)

    axins.set_xlim(-0.5, 2.5)
    axins.set_ylim(-zoom_max - zoom_pad, zoom_max + zoom_pad)
    axins.set_xticks([0, 1, 2])
    axins.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER[:3]], fontsize=8)
    axins.tick_params(axis="y", labelsize=8)
    axins.set_title("zoom: first three models", fontsize=8, pad=8)

    # Draw connectors around the zoomed x-region on the parent axis.
    mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.8)

    # --------------------------------------------------------
    # Panel C
    # --------------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax3.set_ylim(0, 1)

    ax3.annotate(
        "",
        xy=(len(MODEL_ORDER) - 0.55, 0.5),
        xytext=(-0.45, 0.5),
        arrowprops=dict(arrowstyle="->", linewidth=1.6),
    )
    ax3.text(
        -0.45,
        0.75,
        "weak concentration penalty",
        ha="left",
        va="center",
        fontsize=10,
    )
    ax3.text(
        len(MODEL_ORDER) - 0.55,
        0.75,
        "strong hotspot penalty",
        ha="right",
        va="center",
        fontsize=10,
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax3.tick_params(axis="x", length=0)
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Dissipation-model stress test of thermodynamic surrogate ranking",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.35)

    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUT_FILE


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find ranking table at: {INPUT_CSV}\n"
            f"Make sure thermo_ranking_table.csv exists in the thermo_stress_test directory."
        )

    df = load_ranking_table(INPUT_CSV)
    out_path = plot_summary_figure(df)
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
