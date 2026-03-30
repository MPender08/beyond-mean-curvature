from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# Configuration
# =====================================================================
SEEDS = [42, 43, 44, 45, 46]
EXPERIMENT_NAME = "targeted_vs_random"
RESULTS_DIR = Path("results/selective_tail")
OUT_DIR = RESULTS_DIR / "aggregated" / "thermo_stress_test"

# Thermodynamic Scaling Weights
# These map the raw graph metrics into a shared visual "energy" scale.
W_ROUTING = 1.0
W_MAINTENANCE = 5.0
W_DISSIPATION = 0.001

# Ordered so exponent sweeps stay visually/logically consistent.
DISSIPATION_MODELS = OrderedDict({
    "sum_load": {"kind": "power", "alpha": 1.0, "label": r"$\sum load$"},
    "sum_load_1p5": {"kind": "power", "alpha": 1.5, "label": r"$\sum load^{1.5}$"},
    "sum_load_sq": {"kind": "power", "alpha": 2.0, "label": r"$\sum load^2$"},
    "sum_load_cu": {"kind": "power", "alpha": 3.0, "label": r"$\sum load^3$"},
    "max_load": {"kind": "max", "label": r"$\max(load)$"},
    "std_load": {"kind": "std", "label": r"$std(load)$"},
})


# =====================================================================
# Helpers
# =====================================================================
def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_dissipation(edge_loads: list[dict], model: dict) -> float:
    """
    Compute dissipation under one candidate burden model.

    Parameters
    ----------
    edge_loads:
        JSON payload entries of the form:
            [{"edge": [...], "load": int}, ...]
    model:
        One entry from DISSIPATION_MODELS.

    Returns
    -------
    float
        Raw dissipation score before global scaling.
    """
    loads = np.asarray([edge["load"] for edge in edge_loads], dtype=float)

    if loads.size == 0:
        return 0.0

    kind = model["kind"]

    if kind == "power":
        alpha = float(model["alpha"])
        return float(np.sum(loads ** alpha))

    if kind == "max":
        return float(np.max(loads))

    if kind == "std":
        return float(np.std(loads))

    raise ValueError(f"Unknown dissipation model kind: {kind}")


# =====================================================================
# Data loading / summaries
# =====================================================================
def load_thermodynamic_data(dissipation_model_name: str) -> pd.DataFrame:
    """
    Crawl existing CSV/JSON outputs and calculate thermodynamic components
    for each condition and seed under one dissipation model.
    """
    if dissipation_model_name not in DISSIPATION_MODELS:
        raise KeyError(f"Unknown dissipation model: {dissipation_model_name}")

    rows = []
    model = DISSIPATION_MODELS[dissipation_model_name]

    for seed in SEEDS:
        csv_path = RESULTS_DIR / f"seed_{seed}" / "csv" / f"{EXPERIMENT_NAME}.csv"
        json_path = RESULTS_DIR / f"seed_{seed}" / "json" / f"{EXPERIMENT_NAME}_edge_payloads.json"

        if not csv_path.exists() or not json_path.exists():
            print(f"Skipping seed {seed} (missing files)")
            continue

        df_csv = pd.read_csv(csv_path)

        with open(json_path, "r", encoding="utf-8") as f:
            edge_payloads = json.load(f)

        payload_dict = {item["condition_name"]: item for item in edge_payloads}

        for _, row in df_csv.iterrows():
            cond = row["condition_name"]

            if cond not in payload_dict:
                print(f"Skipping condition {cond} for seed {seed} (missing JSON payload)")
                continue

            edge_loads = payload_dict[cond].get("edge_load", [])
            p_diss = compute_dissipation(edge_loads, model)

            rows.append({
                "seed": seed,
                "condition": cond,
                "n_active": int(row["n_active"]),
                "placement": str(row["placement_mode"]),
                "p_routing_raw": float(row["mean_transport_cost"]),
                "p_maint_raw": float(row["maintenance_cost"]),
                "p_diss_raw": float(p_diss),
                "dissipation_model": dissipation_model_name,
                "dissipation_label": model["label"],
            })

    return pd.DataFrame(rows)


def summarize_thermo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average thermodynamic components across seeds and compute the
    scaled thermodynamic surrogate objective.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["placement", "n_active", "dissipation_model", "dissipation_label"])[
            ["p_routing_raw", "p_maint_raw", "p_diss_raw"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    grouped["p_routing"] = grouped["p_routing_raw_mean"] * W_ROUTING
    grouped["p_maint"] = grouped["p_maint_raw_mean"] * W_MAINTENANCE
    grouped["p_diss"] = grouped["p_diss_raw_mean"] * W_DISSIPATION
    grouped["j_thermo"] = grouped["p_routing"] + grouped["p_maint"] + grouped["p_diss"]

    return grouped.sort_values(["placement", "n_active"]).reset_index(drop=True)


# =====================================================================
# Plotting
# =====================================================================
def plot_thermodynamic_curves(summary_df: pd.DataFrame, placement_type: str, model_name: str) -> None:
    """
    Plot intersecting thermodynamic curves for one placement type under
    one dissipation model.
    """
    df_filtered = summary_df[summary_df["placement"] == placement_type].copy()
    if df_filtered.empty:
        print(f"No rows found for placement={placement_type}, model={model_name}")
        return

    df_filtered = df_filtered.sort_values("n_active")
    x = df_filtered["n_active"].to_numpy(dtype=int)

    p_rout = df_filtered["p_routing"].to_numpy(dtype=float)
    p_maint = df_filtered["p_maint"].to_numpy(dtype=float)
    p_diss = df_filtered["p_diss"].to_numpy(dtype=float)
    j_thermo = df_filtered["j_thermo"].to_numpy(dtype=float)

    diss_label = str(df_filtered["dissipation_label"].iloc[0])

    plt.figure(figsize=(8, 6))
    plt.plot(x, p_rout, linestyle="--", linewidth=2, label=r"$P_{routing}$ (Landauer Tax)")
    plt.plot(x, p_maint, linestyle="--", linewidth=2, label=r"$P_{maint}$ (ATP Gate Tax)")
    plt.plot(x, p_diss, linestyle="--", linewidth=2, label=fr"$P_{{dissipation}}$ ({diss_label})")
    plt.plot(x, j_thermo, linewidth=3, label=r"$J_{thermo}$ (Total Energy)")

    min_idx = int(np.argmin(j_thermo))
    plt.plot(x[min_idx], j_thermo[min_idx], marker="o", markersize=10)
    plt.annotate(
        "Thermodynamic\nOptimum",
        xy=(x[min_idx], j_thermo[min_idx]),
        xytext=(x[min_idx] - 1.5, j_thermo[min_idx] + max(j_thermo) * 0.08),
        arrowprops=dict(shrink=0.05, width=1, headwidth=6),
    )

    plt.title(
        f"Thermodynamic Isomorphism: {placement_type.capitalize()} Placement\n"
        f"Dissipation model = {model_name}",
        fontsize=14,
    )
    plt.xlabel("Number of Active Distal Shortcuts (Sparsity Regime)", fontsize=12)
    plt.ylabel("Relative Metabolic Power (Arbitrary Units)", fontsize=12)
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()

    out_path = OUT_DIR / f"thermodynamic_curves_{placement_type}_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_model_sweep(summary_by_model: dict[str, pd.DataFrame], placement_type: str) -> None:
    """
    Overlay J_thermo across dissipation models for one placement type.
    Useful for seeing whether model choice changes curve shape or optimum.
    """
    plt.figure(figsize=(8, 6))

    plotted_any = False
    for model_name, summary_df in summary_by_model.items():
        sub = summary_df[summary_df["placement"] == placement_type].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("n_active")
        x = sub["n_active"].to_numpy(dtype=int)
        y = sub["j_thermo"].to_numpy(dtype=float)

        plt.plot(x, y, linewidth=2, marker="o", label=model_name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        print(f"No summary rows available for placement={placement_type}")
        return

    plt.title(
        f"J_thermo Stress Test Across Dissipation Models\n{placement_type.capitalize()} Placement",
        fontsize=14,
    )
    plt.xlabel("Number of Active Distal Shortcuts (Sparsity Regime)", fontsize=12)
    plt.ylabel("J_thermo (Arbitrary Units)", fontsize=12)
    plt.xticks([5, 10, 20])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    out_path = OUT_DIR / f"j_thermo_model_sweep_{placement_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# =====================================================================
# Stress-test reporting
# =====================================================================
def build_ranking_table(summary_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a compact table showing:
    - best n_active for each placement/model
    - best J_thermo for each placement/model
    - targeted-minus-random gaps at matched n_active
    """
    rows = []

    for model_name, summary_df in summary_by_model.items():
        if summary_df.empty:
            continue

        for placement in ["targeted", "random"]:
            sub = summary_df[summary_df["placement"] == placement].copy()
            if sub.empty:
                continue

            best_idx = sub["j_thermo"].idxmin()
            best_row = sub.loc[best_idx]

            rows.append({
                "dissipation_model": model_name,
                "row_type": "placement_optimum",
                "placement": placement,
                "n_active": int(best_row["n_active"]),
                "value": float(best_row["j_thermo"]),
                "note": "best_j_thermo",
            })

        for n in sorted(summary_df["n_active"].unique()):
            t = summary_df[
                (summary_df["placement"] == "targeted") &
                (summary_df["n_active"] == n)
            ]
            r = summary_df[
                (summary_df["placement"] == "random") &
                (summary_df["n_active"] == n)
            ]

            if len(t) != 1 or len(r) != 1:
                continue

            delta = float(t.iloc[0]["j_thermo"] - r.iloc[0]["j_thermo"])
            winner = "targeted" if delta < 0 else "random" if delta > 0 else "tie"

            rows.append({
                "dissipation_model": model_name,
                "row_type": "matched_gap",
                "placement": "targeted_minus_random",
                "n_active": int(n),
                "value": delta,
                "note": winner,
            })

    return pd.DataFrame(rows)


def build_wide_comparison_table(summary_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a wide table for quick inspection:
    rows   -> dissipation model
    cols   -> placement/n_active combinations
    values -> J_thermo
    """
    long_rows = []

    for model_name, summary_df in summary_by_model.items():
        if summary_df.empty:
            continue

        for _, row in summary_df.iterrows():
            long_rows.append({
                "dissipation_model": model_name,
                "column_name": f"{row['placement']}_n{int(row['n_active']):02d}",
                "j_thermo": float(row["j_thermo"]),
            })

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df

    wide = long_df.pivot(index="dissipation_model", columns="column_name", values="j_thermo")
    return wide.reset_index()


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    ensure_output_dir()

    print("Beginning thermodynamic dissipation stress test...")
    print(f"Output directory: {OUT_DIR}")

    summary_by_model: dict[str, pd.DataFrame] = {}

    for model_name in DISSIPATION_MODELS:
        print(f"\nProcessing dissipation model: {model_name}")

        df_raw = load_thermodynamic_data(model_name)
        if df_raw.empty:
            print(f"No data found for model: {model_name}")
            continue

        raw_out = OUT_DIR / f"thermo_raw_{model_name}.csv"
        df_raw.to_csv(raw_out, index=False)

        summary_df = summarize_thermo(df_raw)
        summary_by_model[model_name] = summary_df

        summary_out = OUT_DIR / f"thermo_summary_{model_name}.csv"
        summary_df.to_csv(summary_out, index=False)

        plot_thermodynamic_curves(summary_df, "targeted", model_name)
        plot_thermodynamic_curves(summary_df, "random", model_name)

    if not summary_by_model:
        print("No model summaries were generated. Check directories and input files.")
        return

    ranking_df = build_ranking_table(summary_by_model)
    ranking_out = OUT_DIR / "thermo_ranking_table.csv"
    ranking_df.to_csv(ranking_out, index=False)

    wide_df = build_wide_comparison_table(summary_by_model)
    wide_out = OUT_DIR / "thermo_wide_comparison_table.csv"
    wide_df.to_csv(wide_out, index=False)

    plot_model_sweep(summary_by_model, "targeted")
    plot_model_sweep(summary_by_model, "random")

    print("\nStress test complete.")
    print(f"Saved ranking table: {ranking_out}")
    print(f"Saved wide comparison table: {wide_out}")
    print("Generated per-model summaries and plots for both placement regimes.")


if __name__ == "__main__":
    main()
