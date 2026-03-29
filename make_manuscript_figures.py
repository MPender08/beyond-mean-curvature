from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_RESULTS_DIR = Path("results/selective_tail")
AGG_DIR = BASE_RESULTS_DIR / "aggregated"
MANUSCRIPT_DIR = BASE_RESULTS_DIR / "manuscript_figures"

SINGLE_RUN_SEED = 42
SINGLE_RUN_EXPERIMENT = "targeted_vs_random"

DPI = 300
FIG_EXT = "png"
SAVE_TRANSPARENT = False


# ============================================================
# I/O utilities
# ============================================================

def ensure_output_dir() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def load_summary_table(experiment_name: str) -> pd.DataFrame:
    path = AGG_DIR / f"{experiment_name}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}")
    return pd.read_csv(path)


def load_diff_summary_table(experiment_name: str, metric_name: str) -> pd.DataFrame:
    path = AGG_DIR / f"{experiment_name}_{metric_name}_diff_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing diff summary table: {path}")
    return pd.read_csv(path)


def load_single_run_table(seed: int, experiment_name: str) -> pd.DataFrame:
    path = BASE_RESULTS_DIR / f"seed_{seed}" / "csv" / f"{experiment_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing single-run table: {path}")
    return pd.read_csv(path)


def normalize_edge_payloads(raw: Any) -> dict[str, Any]:
    """
    Normalize saved edge payload JSON into:
        {condition_name: payload_dict}
    Supports several possible serialized structures.
    """
    if isinstance(raw, dict):
        if "edge_level_payloads" in raw:
            items = raw["edge_level_payloads"]

            # Case: list of dicts with explicit condition_name
            if isinstance(items, list) and items and isinstance(items[0], dict):
                normalized = {}
                for item in items:
                    if "condition_name" not in item:
                        raise ValueError("Missing 'condition_name' in edge_level_payloads item.")
                    condition_name = item["condition_name"]
                    payload = {k: v for k, v in item.items() if k != "condition_name"}
                    normalized[condition_name] = payload
                return normalized

            # Case: list of [condition_name, payload]
            if isinstance(items, list):
                normalized = {}
                for item in items:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        condition_name, payload = item
                        normalized[str(condition_name)] = payload
                    else:
                        raise ValueError(
                            f"Unrecognized item inside edge_level_payloads: {item}"
                        )
                return normalized

        # Already normalized dict keyed by condition name
        if all(isinstance(v, dict) for v in raw.values()):
            return raw

    if isinstance(raw, list):
        # Case: list of dicts with condition_name
        if raw and isinstance(raw[0], dict):
            normalized = {}
            for item in raw:
                if "condition_name" not in item:
                    raise ValueError("Missing 'condition_name' in edge payload item.")
                condition_name = item["condition_name"]
                payload = {k: v for k, v in item.items() if k != "condition_name"}
                normalized[condition_name] = payload
            return normalized

        # Case: list of [condition_name, payload]
        normalized = {}
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                condition_name, payload = item
                normalized[str(condition_name)] = payload
            else:
                raise ValueError(f"Unrecognized edge payload list item: {item}")
        return normalized

    raise ValueError("Unrecognized edge payload structure.")


def load_single_run_edge_payloads(seed: int, experiment_name: str) -> dict[str, Any]:
    path = BASE_RESULTS_DIR / f"seed_{seed}" / "json" / f"{experiment_name}_edge_payloads.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing edge payload json: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, filename: str) -> None:
    outpath = MANUSCRIPT_DIR / filename
    fig.savefig(
        outpath,
        dpi=DPI,
        bbox_inches="tight",
        transparent=SAVE_TRANSPARENT,
    )
    print(f"Saved: {outpath}")


# ============================================================
# Styling helpers
# ============================================================

def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def get_condition_order() -> list[str]:
    return [
        "random_n05",
        "targeted_n05",
        "random_n10",
        "targeted_n10",
        "random_n20",
        "targeted_n20",
    ]


def reorder_summary_df(summary_df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    df = summary_df.copy()
    df["condition_name"] = pd.Categorical(df["condition_name"], categories=order, ordered=True)
    df = df.sort_values("condition_name").reset_index(drop=True)
    return df


def parse_edge_key(edge_key: str) -> tuple[int, int]:
    """
    Parse edge keys stored as strings, e.g. '(3, 7)' or '[3, 7]'.
    """
    try:
        value = ast.literal_eval(edge_key)
    except Exception as exc:
        raise ValueError(f"Could not parse edge key: {edge_key}") from exc

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"Invalid edge key format: {edge_key}")

    return int(value[0]), int(value[1])


# ============================================================
# Generic plotting helpers
# ============================================================

def plot_summary_bar_with_error(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str | None = None,
    order: list[str] | None = None,
) -> None:
    if order is not None:
        summary_df = reorder_summary_df(summary_df, order)

    x = summary_df["condition_name"].astype(str)
    y = summary_df[f"{metric}_mean"]
    yerr = summary_df[f"{metric}_std"]

    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axes(ax)


def plot_diff_bar_with_error(
    ax: plt.Axes,
    diff_summary_df: pd.DataFrame,
    ylabel: str,
    title: str | None = None,
    zero_line: bool = True,
) -> None:
    x = diff_summary_df["n_active"].astype(str)
    y = diff_summary_df["mean"]
    yerr = diff_summary_df["std"]

    ax.bar(x, y, yerr=yerr, capsize=4)
    if zero_line:
        ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("n_active")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=10)
    style_axes(ax)


def plot_single_run_metric_bar(
    ax: plt.Axes,
    run_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    order: list[str] | None = None,
) -> None:
    df = run_df.copy()
    if order is not None:
        df["condition_name"] = pd.Categorical(df["condition_name"], categories=order, ordered=True)
        df = df.sort_values("condition_name").reset_index(drop=True)

    x = df["condition_name"].astype(str)
    y = df[metric]

    ax.bar(x, y)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axes(ax)


def plot_edge_load_vs_curvature_panel(
    ax: plt.Axes,
    edge_payloads: dict[str, Any],
    condition_name: str,
    title: str,
) -> None:
    """
    Plot edge load vs curvature for one condition.

    Supports payload fields stored as:
      - edge_curvatures: list of {"edge": [u, v], "curvature": x}
      - edge_load: list of {"edge": [u, v], "load": y}
      - tail_edges: list of [u, v]
    """
    if condition_name not in edge_payloads:
        raise KeyError(f"Missing condition in edge payloads: {condition_name}")

    payload = edge_payloads[condition_name]

    edge_curvatures_raw = payload["edge_curvatures"]
    edge_load_raw = payload["edge_load"]
    tail_edges_raw = payload["tail_edges"]

    edge_curvatures: dict[tuple[int, int], float] = {}
    if isinstance(edge_curvatures_raw, dict):
        for k, v in edge_curvatures_raw.items():
            edge = parse_edge_key(k) if isinstance(k, str) else tuple(k)
            edge_curvatures[edge] = float(v)
    elif isinstance(edge_curvatures_raw, list):
        for item in edge_curvatures_raw:
            edge = tuple(item["edge"])
            edge_curvatures[(int(edge[0]), int(edge[1]))] = float(item["curvature"])
    else:
        raise ValueError("Unsupported edge_curvatures structure.")

    edge_load: dict[tuple[int, int], float] = {}
    if isinstance(edge_load_raw, dict):
        for k, v in edge_load_raw.items():
            edge = parse_edge_key(k) if isinstance(k, str) else tuple(k)
            edge_load[edge] = float(v)
    elif isinstance(edge_load_raw, list):
        for item in edge_load_raw:
            edge = tuple(item["edge"])
            edge_load[(int(edge[0]), int(edge[1]))] = float(item["load"])
    else:
        raise ValueError("Unsupported edge_load structure.")

    tail_edges: set[tuple[int, int]] = set()
    for e in tail_edges_raw:
        if isinstance(e, str):
            tail_edges.add(parse_edge_key(e))
        else:
            tail_edges.add((int(e[0]), int(e[1])))

    edges = sorted(edge_curvatures.keys())
    curvatures = np.array([edge_curvatures[e] for e in edges], dtype=float)
    loads = np.array([edge_load.get(e, 0.0) for e in edges], dtype=float)
    is_tail = np.array([e in tail_edges for e in edges], dtype=bool)

    ax.scatter(curvatures[~is_tail], loads[~is_tail], alpha=0.7, label="non-tail edges")
    ax.scatter(curvatures[is_tail], loads[is_tail], alpha=0.9, label="q10 tail edges")
    ax.set_xlabel("Edge curvature")
    ax.set_ylabel("Transport load")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)


# ============================================================
# Figure 1 custom schematic helpers
# ============================================================

def draw_base_graph_schematic(ax: plt.Axes) -> None:
    """
    Panel A for Figure 1.
    Draw a simple balanced tree with a few distal shortcuts highlighted.
    """
    G = nx.balanced_tree(r=2, h=3)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        # Manual tree-like fallback layout
        root = 0
        lengths = nx.single_source_shortest_path_length(G, root)

        levels: dict[int, list[int]] = {}
        for node, depth in lengths.items():
            levels.setdefault(depth, []).append(node)

        pos = {}
        max_depth = max(levels)
        for depth, nodes in levels.items():
            nodes = sorted(nodes)
            xs = np.linspace(-1.0, 1.0, len(nodes))
            y = max_depth - depth
            for x, node in zip(xs, nodes):
                pos[node] = (x, y)

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80)

    shortcut_edges = [(7, 10), (8, 13), (9, 12)]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=shortcut_edges,
        ax=ax,
        width=2.0,
        style="dashed",
    )

    ax.set_title("Base hierarchy + distal shortcuts", fontsize=10)
    ax.axis("off")

def draw_curvature_distribution_schematic(ax: plt.Axes) -> None:
    """
    Panel B for Figure 1.
    Stylized distribution showing mean vs q10 tail.
    """
    x = np.linspace(-1.0, 1.0, 400)
    y = np.exp(-((x + 0.1) ** 2) / 0.12) + 0.25 * np.exp(-((x + 0.7) ** 2) / 0.02)
    y = y / y.max()

    ax.plot(x, y, linewidth=2)
    q10_x = -0.65
    mean_x = -0.05

    ax.axvline(mean_x, linestyle="--", linewidth=1.5)
    ax.axvline(q10_x, linestyle="--", linewidth=1.5)
    ax.fill_between(x, y, where=(x <= q10_x), alpha=0.25)

    ax.text(mean_x, 1.02, "mean", ha="center", va="bottom", fontsize=9)
    ax.text(q10_x, 1.02, "q10", ha="center", va="bottom", fontsize=9)

    ax.set_title("Curvature distribution: mean vs lower tail", fontsize=10)
    ax.set_xlabel("Edge curvature")
    ax.set_ylabel("Density")
    style_axes(ax)


def draw_distributed_tail_schematic(ax: plt.Axes) -> None:
    """
    Panel C for Figure 1.
    Cartoon showing broad tail recruitment.
    """
    ax.set_title("Distributed tail recruitment", fontsize=10)

    xs = np.linspace(0.1, 0.9, 8)
    ys = np.full_like(xs, 0.5)
    ax.scatter(xs, ys, s=120)

    # Highlight several recruited tail edges
    for i in [1, 2, 4, 5]:
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], linewidth=3)

    ax.text(0.5, 0.2, "broader lower-tail participation", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_concentrated_tail_schematic(ax: plt.Axes) -> None:
    """
    Panel D for Figure 1.
    Cartoon showing concentrated tail exploitation.
    """
    ax.set_title("Concentrated tail exploitation", fontsize=10)

    xs = np.linspace(0.1, 0.9, 8)
    ys = np.full_like(xs, 0.5)
    ax.scatter(xs, ys, s=120)

    # Highlight one or two heavily used tail edges
    ax.plot([xs[3], xs[4]], [ys[3], ys[4]], linewidth=5)
    ax.plot([xs[4], xs[5]], [ys[4], ys[5]], linewidth=5)

    ax.text(0.5, 0.2, "hotter per-edge tail burden", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ============================================================
# Figure 5 custom helpers
# ============================================================

def plot_regime_summary_panel(
    ax: plt.Axes,
    tail_diff_df: pd.DataFrame,
    obj_diff_df: pd.DataFrame,
    tbc_diff_df: pd.DataFrame,
    n_active: int,
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Compact summary panel for one regime.

    Uses three bars:
      - tail_load_fraction diff
      - objective_J diff
      - tail_burden_concentration diff
    """
    tail_row = tail_diff_df.loc[tail_diff_df["n_active"] == n_active].iloc[0]
    obj_row = obj_diff_df.loc[obj_diff_df["n_active"] == n_active].iloc[0]
    tbc_row = tbc_diff_df.loc[tbc_diff_df["n_active"] == n_active].iloc[0]

    labels = ["TLF diff", "J diff", "TBC diff"]
    values = [tail_row["mean"], obj_row["mean"], tbc_row["mean"]]
    errors = [tail_row["std"], obj_row["std"], tbc_row["std"]]

    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10, pad=6)

    if subtitle is not None:
        ax.text(
            0.5,
            0.98,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.08 if ymax > 0 else ymax)
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)


def draw_regime_summary_cartoon(ax: plt.Axes) -> None:
    ax.set_title("Schematic summary", fontsize=10)
    ax.text(0.5, 0.72, "targeted = broader recruited tail", ha="center", fontsize=10)
    ax.text(0.5, 0.42, "random = more concentrated exploited tail", ha="center", fontsize=10)
    ax.text(0.5, 0.15, "objective depends on tradeoff, not one metric", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ============================================================
# Figure builders
# ============================================================

def build_figure1_conceptual_framing() -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    draw_base_graph_schematic(axs[0, 0])
    add_panel_label(axs[0, 0], "A")

    draw_curvature_distribution_schematic(axs[0, 1])
    add_panel_label(axs[0, 1], "B")

    draw_distributed_tail_schematic(axs[1, 0])
    add_panel_label(axs[1, 0], "C")

    draw_concentrated_tail_schematic(axs[1, 1])
    add_panel_label(axs[1, 1], "D")

    fig.suptitle("From mean curvature to lower-tail routing structure", fontsize=12)
    fig.tight_layout()
    return fig


def build_figure2_single_run_proof_of_concept(
    seed: int = SINGLE_RUN_SEED,
    experiment_name: str = SINGLE_RUN_EXPERIMENT,
) -> plt.Figure:
    run_df = load_single_run_table(seed, experiment_name)
    edge_payloads_raw = load_single_run_edge_payloads(seed, experiment_name)
    edge_payloads = normalize_edge_payloads(edge_payloads_raw)
    order = get_condition_order()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_single_run_metric_bar(
        axs[0, 0],
        run_df,
        metric="objective_J",
        ylabel="Objective J",
        title="Single-run objective",
        order=order,
    )
    add_panel_label(axs[0, 0], "A")

    plot_single_run_metric_bar(
        axs[0, 1],
        run_df,
        metric="tail_load_fraction",
        ylabel="Tail load fraction",
        title="Single-run tail participation",
        order=order,
    )
    add_panel_label(axs[0, 1], "B")

    plot_edge_load_vs_curvature_panel(
        axs[1, 0],
        edge_payloads=edge_payloads,
        condition_name="targeted_n10",
        title="targeted_n10: load vs curvature",
    )
    add_panel_label(axs[1, 0], "C")

    plot_edge_load_vs_curvature_panel(
        axs[1, 1],
        edge_payloads=edge_payloads,
        condition_name="random_n10",
        title="random_n10: load vs curvature",
    )
    add_panel_label(axs[1, 1], "D")

    fig.suptitle("Sparse lower-tail structure shapes transport differently than random placement", fontsize=12)
    fig.tight_layout()
    return fig


def build_figure3_multiseed_aggregation(
    experiment_name: str = "targeted_vs_random",
) -> plt.Figure:
    summary_df = load_summary_table(experiment_name)
    order = get_condition_order()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_summary_bar_with_error(
        axs[0, 0],
        summary_df,
        metric="objective_J",
        ylabel="Objective J",
        title="Objective J",
        order=order,
    )
    add_panel_label(axs[0, 0], "A")

    plot_summary_bar_with_error(
        axs[0, 1],
        summary_df,
        metric="tail_load_fraction",
        ylabel="Tail load fraction",
        title="Tail load fraction",
        order=order,
    )
    add_panel_label(axs[0, 1], "B")

    plot_summary_bar_with_error(
        axs[1, 0],
        summary_df,
        metric="mean_transport_cost",
        ylabel="Mean transport cost",
        title="Mean transport cost",
        order=order,
    )
    add_panel_label(axs[1, 0], "C")

    plot_summary_bar_with_error(
        axs[1, 1],
        summary_df,
        metric="q10_curvature",
        ylabel="q10 curvature",
        title="q10 curvature",
        order=order,
    )
    add_panel_label(axs[1, 1], "D")

    fig.suptitle("Lower-tail effects are regime-dependent across repeated seeds", fontsize=12)
    fig.tight_layout()
    return fig


def build_figure4_tail_organization_metrics(
    experiment_name: str = "targeted_vs_random",
) -> plt.Figure:
    summary_df = load_summary_table(experiment_name)
    order = get_condition_order()

    tail_diff_df = load_diff_summary_table(experiment_name, "tail_load_fraction")
    obj_diff_df = load_diff_summary_table(experiment_name, "objective_J")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_summary_bar_with_error(
        axs[0, 0],
        summary_df,
        metric="tail_efficiency_ratio",
        ylabel="Tail efficiency ratio",
        title="Tail efficiency ratio",
        order=order,
    )
    add_panel_label(axs[0, 0], "A")

    plot_summary_bar_with_error(
        axs[0, 1],
        summary_df,
        metric="tail_burden_concentration",
        ylabel="Tail burden concentration",
        title="Tail burden concentration",
        order=order,
    )
    add_panel_label(axs[0, 1], "B")

    plot_diff_bar_with_error(
        axs[1, 0],
        tail_diff_df,
        ylabel="targeted - random",
        title="Tail load fraction difference",
    )
    add_panel_label(axs[1, 0], "C")

    plot_diff_bar_with_error(
        axs[1, 1],
        obj_diff_df,
        ylabel="targeted - random",
        title="Objective J difference",
    )
    add_panel_label(axs[1, 1], "D")

    fig.suptitle("Tail organization decomposes into participation and burden concentration", fontsize=12)
    fig.tight_layout()
    return fig


def build_figure5_regime_diagram(
    experiment_name: str = "targeted_vs_random",
) -> plt.Figure:
    tail_diff_df = load_diff_summary_table(experiment_name, "tail_load_fraction")
    obj_diff_df = load_diff_summary_table(experiment_name, "objective_J")
    tbc_diff_df = load_diff_summary_table(experiment_name, "tail_burden_concentration")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_regime_summary_panel(
        axs[0, 0],
        tail_diff_df=tail_diff_df,
        obj_diff_df=obj_diff_df,
        tbc_diff_df=tbc_diff_df,
        n_active=5,
        title="Sparse regime (n05)",
        subtitle="recruitment clearest",
    )
    add_panel_label(axs[0, 0], "A")

    plot_regime_summary_panel(
        axs[0, 1],
        tail_diff_df=tail_diff_df,
        obj_diff_df=obj_diff_df,
        tbc_diff_df=tbc_diff_df,
        n_active=10,
        title="Intermediate regime (n10)",
        subtitle="mixed / transition",
    )
    add_panel_label(axs[0, 1], "B")

    plot_regime_summary_panel(
        axs[1, 0],
        tail_diff_df=tail_diff_df,
        obj_diff_df=obj_diff_df,
        tbc_diff_df=tbc_diff_df,
        n_active=20,
        title="Saturated regime (n20)",
        subtitle="participation fades, concentration persists",
    )
    add_panel_label(axs[1, 0], "C")

    draw_regime_summary_cartoon(axs[1, 1])
    add_panel_label(axs[1, 1], "D")

    fig.suptitle("Lower-tail organization changes across sparsity regimes", fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================
# Main driver
# ============================================================

def main() -> None:
    ensure_output_dir()

    fig1 = build_figure1_conceptual_framing()
    save_figure(fig1, f"figure1_conceptual_framing.{FIG_EXT}")
    plt.close(fig1)

    fig2 = build_figure2_single_run_proof_of_concept()
    save_figure(fig2, f"figure2_single_run_proof_of_concept.{FIG_EXT}")
    plt.close(fig2)

    fig3 = build_figure3_multiseed_aggregation()
    save_figure(fig3, f"figure3_multiseed_aggregation.{FIG_EXT}")
    plt.close(fig3)

    fig4 = build_figure4_tail_organization_metrics()
    save_figure(fig4, f"figure4_tail_organization_metrics.{FIG_EXT}")
    plt.close(fig4)

    fig5 = build_figure5_regime_diagram()
    save_figure(fig5, f"figure5_regime_diagram.{FIG_EXT}")
    plt.close(fig5)


if __name__ == "__main__":
    main()