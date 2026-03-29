from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Helpers
# ============================================================

def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _results_to_dict(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Convert flat result rows into a condition_name -> row lookup.
    """
    return {row["condition_name"]: row for row in results}


def _edge_payloads_to_dict(
    edge_level_payloads: list[tuple[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """
    Convert edge payloads into a condition_name -> payload lookup.
    """
    return {condition_name: payload for condition_name, payload in edge_level_payloads}


# ============================================================
# Core plots
# ============================================================

def plot_condition_metric_comparison(
    results: list[dict[str, Any]],
    metric: str,
    title: str | None = None,
    ylabel: str | None = None,
    savepath: str | Path | None = None,
) -> None:
    """
    Simple bar plot for one scalar metric across conditions.
    """
    condition_names = [row["condition_name"] for row in results]
    values = [row.get(metric, np.nan) for row in results]

    plt.figure(figsize=(8, 4.5))
    plt.bar(condition_names, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel if ylabel is not None else metric)
    plt.title(title if title is not None else f"{metric} across conditions")
    plt.tight_layout()

    if savepath is not None:
        ensure_parent_dir(savepath)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_edge_load_vs_curvature(
    payload: dict[str, Any],
    condition_name: str,
    title: str | None = None,
    savepath: str | Path | None = None,
) -> None:
    """
    Scatter plot of per-edge transport load vs edge curvature, highlighting q-tail edges.
    """
    edge_payloads = _edge_payloads_to_dict(payload["edge_level_payloads"])
    if condition_name not in edge_payloads:
        raise KeyError(f"Condition '{condition_name}' not found in edge_level_payloads")

    data = edge_payloads[condition_name]
    edge_curvatures = data["edge_curvatures"]
    edge_load = data["edge_load"]
    tail_edges = data["tail_edges"]

    edges = sorted(edge_curvatures.keys())
    curvatures = np.array([edge_curvatures[e] for e in edges], dtype=float)
    loads = np.array([edge_load.get(e, 0) for e in edges], dtype=float)
    is_tail = np.array([e in tail_edges for e in edges], dtype=bool)

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(
        curvatures[~is_tail],
        loads[~is_tail],
        alpha=0.7,
        label="non-tail edges",
    )
    plt.scatter(
        curvatures[is_tail],
        loads[is_tail],
        alpha=0.9,
        label="q-tail edges",
    )
    plt.xlabel("Edge curvature")
    plt.ylabel("Transport load")
    plt.title(title if title is not None else f"Load vs curvature: {condition_name}")
    plt.legend(frameon=False)
    plt.tight_layout()

    if savepath is not None:
        ensure_parent_dir(savepath)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_transport_vs_curvature_summary(
    results: list[dict[str, Any]],
    x_metric: str = "q10_curvature",
    y_metric: str = "objective_J",
    title: str | None = None,
    savepath: str | Path | None = None,
) -> None:
    """
    Scatter plot of condition-level summary metrics.
    """
    x = np.array([row.get(x_metric, np.nan) for row in results], dtype=float)
    y = np.array([row.get(y_metric, np.nan) for row in results], dtype=float)
    labels = [row["condition_name"] for row in results]

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(x, y)

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), fontsize=8, alpha=0.8)

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title if title is not None else f"{y_metric} vs {x_metric}")
    plt.tight_layout()

    if savepath is not None:
        ensure_parent_dir(savepath)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# Experiment-specific convenience plots
# ============================================================

def plot_targeted_vs_random_summary(
    results: list[dict[str, Any]],
    save_dir: str | Path | None = None,
) -> None:
    """
    Convenience bundle for targeted_vs_random runs.
    """
    save_dir = Path(save_dir) if save_dir is not None else None

    plot_condition_metric_comparison(
        results,
        metric="objective_J",
        title="Objective J across targeted vs random conditions",
        ylabel="Objective J",
        savepath=(save_dir / "targeted_vs_random_objective_J.png") if save_dir else None,
    )

    plot_condition_metric_comparison(
        results,
        metric="tail_load_fraction",
        title="Tail load fraction across targeted vs random conditions",
        ylabel="Tail load fraction",
        savepath=(save_dir / "targeted_vs_random_tail_load_fraction.png") if save_dir else None,
    )

    plot_condition_metric_comparison(
        results,
        metric="mean_transport_cost",
        title="Mean transport cost across targeted vs random conditions",
        ylabel="Mean transport cost",
        savepath=(save_dir / "targeted_vs_random_mean_transport_cost.png") if save_dir else None,
    )


def plot_betweenness_vs_curvature_summary(
    results: list[dict[str, Any]],
    save_dir: str | Path | None = None,
) -> None:
    """
    Convenience bundle for betweenness_vs_curvature runs.
    """
    save_dir = Path(save_dir) if save_dir is not None else None

    plot_condition_metric_comparison(
        results,
        metric="tail_load_fraction",
        title="Tail load fraction across conditions",
        ylabel="Tail load fraction",
        savepath=(save_dir / "betweenness_vs_curvature_tail_load_fraction.png") if save_dir else None,
    )

    plot_condition_metric_comparison(
        results,
        metric="objective_J",
        title="Objective J across conditions",
        ylabel="Objective J",
        savepath=(save_dir / "betweenness_vs_curvature_objective_J.png") if save_dir else None,
    )

    plot_transport_vs_curvature_summary(
        results,
        x_metric="q10_curvature",
        y_metric="objective_J",
        title="Objective J vs lower-tail curvature",
        savepath=(save_dir / "betweenness_vs_curvature_J_vs_q10.png") if save_dir else None,
    )

    plot_transport_vs_curvature_summary(
        results,
        x_metric="mean_curvature",
        y_metric="tail_load_fraction",
        title="Tail load fraction vs mean curvature",
        savepath=(save_dir / "betweenness_vs_curvature_tail_load_vs_mean_curvature.png") if save_dir else None,
    )


# ============================================================
# Master dispatcher
# ============================================================

def make_all_plots(payload: dict[str, Any], config: dict[str, Any]) -> None:
    """
    Main plot dispatcher used by run_selective_tail_experiment.py.

    Expected payload format:
        {
            "results": [flat row dicts...],
            "edge_level_payloads": [(condition_name, payload), ...],
        }
    """
    results = payload["results"]
    if not results:
        print("No results available for plotting.")
        return

    experiment_name = config["name"]
    save_dir = Path(config.get("results_dir", "results/selective_tail")) / "figures" / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generic always-useful plots
    plot_condition_metric_comparison(
        results,
        metric="objective_J",
        title=f"{experiment_name}: Objective J",
        ylabel="Objective J",
        savepath=save_dir / "objective_J.png",
    )

    plot_condition_metric_comparison(
        results,
        metric="mean_transport_cost",
        title=f"{experiment_name}: Mean transport cost",
        ylabel="Mean transport cost",
        savepath=save_dir / "mean_transport_cost.png",
    )

    plot_condition_metric_comparison(
        results,
        metric="tail_load_fraction",
        title=f"{experiment_name}: Tail load fraction",
        ylabel="Tail load fraction",
        savepath=save_dir / "tail_load_fraction.png",
    )

    plot_transport_vs_curvature_summary(
        results,
        x_metric="q10_curvature",
        y_metric="objective_J",
        title=f"{experiment_name}: Objective J vs q10 curvature",
        savepath=save_dir / "J_vs_q10_curvature.png",
    )

    # Experiment-specific bundles
    if experiment_name == "targeted_vs_random":
        plot_targeted_vs_random_summary(results, save_dir=save_dir)

        # Important mechanism plots
        for condition_name in ["targeted_n10", "random_n10", "targeted_n20", "random_n20"]:
            try:
                plot_edge_load_vs_curvature(
                    payload,
                    condition_name=condition_name,
                    title=f"{condition_name}: load vs curvature",
                    savepath=save_dir / f"{condition_name}_load_vs_curvature.png",
                )
            except KeyError:
                pass

    elif experiment_name == "betweenness_vs_curvature":
        plot_betweenness_vs_curvature_summary(results, save_dir=save_dir)

        for condition_name in [
            "uniform_gamma_020",
            "uniform_gamma_040",
            "uniform_gamma_060",
            "uniform_gamma_080",
            "sparse_targeted_n10",
            "sparse_targeted_n20",
        ]:
            try:
                plot_edge_load_vs_curvature(
                    payload,
                    condition_name=condition_name,
                    title=f"{condition_name}: load vs curvature",
                    savepath=save_dir / f"{condition_name}_load_vs_curvature.png",
                )
            except KeyError:
                pass