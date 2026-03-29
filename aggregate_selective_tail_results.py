from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SEEDS = [42, 43, 44, 45, 46]
EXPERIMENT_NAME = "targeted_vs_random"


def load_seed_results(seeds: list[int], experiment_name: str) -> pd.DataFrame:
    dfs = []

    for seed in seeds:
        csv_path = Path(f"results/selective_tail/seed_{seed}/csv/{experiment_name}.csv")
        if not csv_path.exists():
            print(f"Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No seed result CSVs found.")

    return pd.concat(dfs, ignore_index=True)


def aggregate_metrics(
    df: pd.DataFrame,
    metrics: list[str],
    group_col: str = "condition_name",
) -> pd.DataFrame:
    agg_spec = {}
    for metric in metrics:
        agg_spec[metric] = ["mean", "std", "min", "max", "count"]

    grouped = df.groupby(group_col).agg(agg_spec)
    grouped.columns = [
        f"{metric}_{stat}" for metric, stat in grouped.columns
    ]
    grouped = grouped.reset_index()

    return grouped


def compute_paired_differences(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    for seed, df_seed in df.groupby("seed"):
        for n in [5, 10, 20]:
            t_name = f"targeted_n{n:02d}"
            r_name = f"random_n{n:02d}"

            t_row = df_seed[df_seed["condition_name"] == t_name]
            r_row = df_seed[df_seed["condition_name"] == r_name]

            if len(t_row) == 0 or len(r_row) == 0:
                continue

            t_val = float(t_row.iloc[0][metric])
            r_val = float(r_row.iloc[0][metric])

            rows.append({
                "seed": seed,
                "n_active": n,
                f"{metric}_targeted": t_val,
                f"{metric}_random": r_val,
                f"{metric}_diff": t_val - r_val,
            })

    return pd.DataFrame(rows)


def summarize_paired_differences(diff_df: pd.DataFrame, diff_col: str) -> pd.DataFrame:
    """
    Group paired targeted-minus-random differences by n_active.

    Returns columns:
        mean, std, min, max, count
    """
    return (
        diff_df.groupby("n_active")[diff_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


def plot_paired_difference_summary(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    savepath: str | None = None,
) -> None:
    x = summary_df[x_col]
    y = summary_df[y_col]
    yerr = summary_df["std"]

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(x.astype(str), y, yerr=yerr, capsize=5)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("n_active")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_metric_with_errorbars(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    savepath: str | None = None,
) -> None:
    x = summary_df["condition_name"]
    y = summary_df[f"{metric}_mean"]
    yerr = summary_df[f"{metric}_std"]

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, y, yerr=yerr, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    metrics = [
        "objective_J",
        "tail_load_fraction",
        "tail_efficiency_ratio",
        "mean_transport_cost",
        "q10_curvature",
        "tail_burden_concentration",
    ]

    df = load_seed_results(SEEDS, EXPERIMENT_NAME)

    out_dir = Path("results/selective_tail/aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{EXPERIMENT_NAME}_all_seeds.csv", index=False)

    summary_df = aggregate_metrics(df, metrics)
    summary_df.to_csv(out_dir / f"{EXPERIMENT_NAME}_summary.csv", index=False)

    tail_diff_df = compute_paired_differences(df, "tail_load_fraction")
    tail_diff_df.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_tail_load_fraction_diffs.csv",
        index=False,
    )
    tail_diff_summary = summarize_paired_differences(
        tail_diff_df,
        "tail_load_fraction_diff",
    )
    tail_diff_summary.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_tail_load_fraction_diff_summary.csv",
        index=False,
    )
    print("\nTail load fraction paired differences (targeted - random):")
    print(tail_diff_summary)

    obj_diff_df = compute_paired_differences(df, "objective_J")
    obj_diff_df.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_objective_J_diffs.csv",
        index=False,
    )
    obj_diff_summary = summarize_paired_differences(
        obj_diff_df,
        "objective_J_diff",
    )
    obj_diff_summary.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_objective_J_diff_summary.csv",
        index=False,
    )
    print("\nObjective J paired differences (targeted - random):")
    print(obj_diff_summary)

    ter_diff_df = compute_paired_differences(df, "tail_efficiency_ratio")
    ter_diff_df.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_tail_efficiency_ratio_diffs.csv",
        index=False,
    )
    ter_diff_summary = summarize_paired_differences(
        ter_diff_df,
        "tail_efficiency_ratio_diff",
    )
    ter_diff_summary.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_tail_efficiency_ratio_diff_summary.csv",
        index=False,
    )
    print("\nTail efficiency ratio paired differences (targeted - random):")
    print(ter_diff_summary)
    
    tbc_diff_df = compute_paired_differences(df, "tail_burden_concentration")
    tbc_diff_df.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_tail_burden_concentration_diffs.csv",
        index=False,
    )
    tbc_diff_summary = summarize_paired_differences(
        tbc_diff_df,
        "tail_burden_concentration_diff",
    )
    tbc_diff_summary.to_csv(
        out_dir / f"{EXPERIMENT_NAME}_tail_burden_concentration_diff_summary.csv",
        index=False,
    )
    print("\nTail burden concentration paired differences (targeted - random):")
    print(tbc_diff_summary)
    
    print(summary_df)

    plot_metric_with_errorbars(
        summary_df,
        metric="tail_load_fraction",
        title="Tail load fraction across seeds",
        ylabel="Tail load fraction",
        savepath=out_dir / f"{EXPERIMENT_NAME}_tail_load_fraction_errorbars.png",
    )

    plot_metric_with_errorbars(
        summary_df,
        metric="objective_J",
        title="Objective J across seeds",
        ylabel="Objective J",
        savepath=out_dir / f"{EXPERIMENT_NAME}_objective_J_errorbars.png",
    )

    plot_metric_with_errorbars(
        summary_df,
        metric="mean_transport_cost",
        title="Mean transport cost across seeds",
        ylabel="Mean transport cost",
        savepath=out_dir / f"{EXPERIMENT_NAME}_mean_transport_cost_errorbars.png",
    )

    plot_metric_with_errorbars(
        summary_df,
        metric="q10_curvature",
        title="q10 curvature across seeds",
        ylabel="q10 curvature",
        savepath=out_dir / f"{EXPERIMENT_NAME}_q10_curvature_errorbars.png",
    )
   
    plot_metric_with_errorbars(
        summary_df,
        metric="tail_efficiency_ratio",
        title="Tail efficiency ratio across seeds",
        ylabel="Tail efficiency ratio",
        savepath=out_dir / f"{EXPERIMENT_NAME}_tail_efficiency_ratio_errorbars.png",
)

    plot_paired_difference_summary(
        tail_diff_summary,
        x_col="n_active",
        y_col="mean",
        title="Tail load fraction difference by edge budget",
        ylabel="targeted - random (positive favors targeted)",
        savepath=out_dir / f"{EXPERIMENT_NAME}_tail_load_fraction_diff_summary.png",
    )

    plot_paired_difference_summary(
        obj_diff_summary,
        x_col="n_active",
        y_col="mean",
        title="Objective J difference by edge budget",
        ylabel="targeted - random (negative favors targeted)",
        savepath=out_dir / f"{EXPERIMENT_NAME}_objective_J_diff_summary.png",
    )

    plot_paired_difference_summary(
        ter_diff_summary,
        x_col="n_active",
        y_col="mean",
        title="Tail efficiency ratio difference by edge budget",
        ylabel="targeted - random (positive favors targeted)",
        savepath=out_dir / f"{EXPERIMENT_NAME}_tail_efficiency_ratio_diff_summary.png",
    )

    plot_metric_with_errorbars(
        summary_df,
        metric="tail_burden_concentration",
        title="Tail burden concentration across seeds",
        ylabel="Tail burden concentration",
        savepath=out_dir / f"{EXPERIMENT_NAME}_tail_burden_concentration_errorbars.png",
    )

    plot_paired_difference_summary(
        tbc_diff_summary,
        x_col="n_active",
        y_col="mean",
        title="Tail burden concentration difference by edge budget",
        ylabel="targeted - random (positive favors targeted)",
        savepath=out_dir / f"{EXPERIMENT_NAME}_tail_burden_concentration_diff_summary.png",
    )



if __name__ == "__main__":
    main()