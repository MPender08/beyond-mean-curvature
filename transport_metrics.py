from __future__ import annotations

from collections import Counter
from typing import Any

import math
import numpy as np
import networkx as nx


Edge = tuple[int, int]


# ============================================================
# Helpers
# ============================================================

def normalize_edge(u: int, v: int) -> Edge:
    """
    Store edges in canonical sorted order.
    """
    return (u, v) if u <= v else (v, u)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


# ============================================================
# Transport simulation
# ============================================================

def simulate_transport(
    G: nx.Graph,
    n_pairs: int = 1000,
    weighted: bool = True,
    rng: np.random.Generator | None = None,
    weight_attr: str = "weight",
) -> dict[str, Any]:
    """
    Simulate source-target routing across the graph.

    Parameters
    ----------
    G:
        Graph to route over.
    n_pairs:
        Number of random source-target pairs to sample.
    weighted:
        If True, use weighted shortest paths. Otherwise use unweighted paths.
    rng:
        Random number generator.
    weight_attr:
        Edge attribute used for weighted shortest paths.

    Returns
    -------
    dict
        {
            "pair_costs": list[float],
            "pair_hops": list[int],
            "edge_load": Counter[Edge],
            "mean_transport_cost": float,
            "median_transport_cost": float,
            "transport_cost_std": float,
            "mean_hops": float,
            "median_hops": float,
            "max_edge_load": int,
            "load_std": float,
        }
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nodes = np.asarray(list(G.nodes()))
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least 2 nodes for transport simulation.")

    pair_costs: list[float] = []
    pair_hops: list[int] = []
    edge_load: Counter[Edge] = Counter()

    path_weight = weight_attr if weighted else None

    for _ in range(n_pairs):
        s, t = rng.choice(nodes, size=2, replace=False)

        path = nx.shortest_path(G, source=int(s), target=int(t), weight=path_weight)

        cost = 0.0
        hops = max(len(path) - 1, 0)

        for a, b in zip(path[:-1], path[1:]):
            if weighted:
                cost += float(G[a][b].get(weight_attr, 1.0))
            else:
                cost += 1.0

            edge = normalize_edge(a, b)
            edge_load[edge] += 1

        pair_costs.append(cost)
        pair_hops.append(hops)

    pair_costs_arr = np.asarray(pair_costs, dtype=float)
    pair_hops_arr = np.asarray(pair_hops, dtype=float)
    edge_load_arr = np.asarray(list(edge_load.values()), dtype=float) if edge_load else np.asarray([], dtype=float)

    return {
        "pair_costs": pair_costs,
        "pair_hops": pair_hops,
        "edge_load": edge_load,
        "mean_transport_cost": _safe_mean(pair_costs_arr),
        "median_transport_cost": _safe_median(pair_costs_arr),
        "transport_cost_std": _safe_std(pair_costs_arr),
        "mean_hops": _safe_mean(pair_hops_arr),
        "median_hops": _safe_median(pair_hops_arr),
        "max_edge_load": int(np.max(edge_load_arr)) if edge_load_arr.size else 0,
        "load_std": _safe_std(edge_load_arr),
    }


# ============================================================
# Graph-level metrics
# ============================================================

def compute_graph_metrics(
    G: nx.Graph,
    weight_attr: str = "weight",
) -> dict[str, float]:
    """
    Compute graph-wide routing / integration metrics.

    Returns
    -------
    dict
        {
            "mean_shortest_path": float,
            "global_efficiency": float,
            "edge_betweenness_max": float,
            "edge_betweenness_mean": float,
            "edge_betweenness_std": float,
        }
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return {
            "mean_shortest_path": float("nan"),
            "global_efficiency": float("nan"),
            "edge_betweenness_max": float("nan"),
            "edge_betweenness_mean": float("nan"),
            "edge_betweenness_std": float("nan"),
        }

    # Weighted all-pairs shortest path
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight_attr))
    all_lengths = []

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            all_lengths.append(lengths[u][v])

    all_lengths_arr = np.asarray(all_lengths, dtype=float)

    # NOTE:
    # nx.global_efficiency ignores edge weights.
    # We keep it here as a classical topology proxy and add weighted shortest path separately.
    edge_betweenness = nx.edge_betweenness_centrality(G, weight=weight_attr)
    edge_bet_vals = np.asarray(list(edge_betweenness.values()), dtype=float) if edge_betweenness else np.asarray([], dtype=float)

    return {
        "mean_shortest_path": _safe_mean(all_lengths_arr),
        "global_efficiency": float(nx.global_efficiency(G)),
        "edge_betweenness_max": float(np.max(edge_bet_vals)) if edge_bet_vals.size else float("nan"),
        "edge_betweenness_mean": _safe_mean(edge_bet_vals),
        "edge_betweenness_std": _safe_std(edge_bet_vals),
    }


# ============================================================
# Tail / non-tail transport diagnostics
# ============================================================

def compute_tail_load_metrics(
    edge_load: dict[Edge, int] | Counter[Edge],
    tail_edges: set[Edge],
) -> dict[str, float]:
    if not edge_load:
        return {
            "tail_edge_load_mean": 0.0,
            "non_tail_edge_load_mean": 0.0,
            "tail_load_fraction": 0.0,
            "tail_edge_fraction_loaded": 0.0,
            "tail_efficiency_ratio": 0.0,
            "tail_burden_concentration": 0.0,
        }

    tail_loads = []
    non_tail_loads = []

    total_load = 0.0
    tail_total_load = 0.0

    for edge, load in edge_load.items():
        total_load += load
        if edge in tail_edges:
            tail_loads.append(load)
            tail_total_load += load
        else:
            non_tail_loads.append(load)

    tail_arr = np.asarray(tail_loads, dtype=float) if tail_loads else np.asarray([], dtype=float)
    non_tail_arr = np.asarray(non_tail_loads, dtype=float) if non_tail_loads else np.asarray([], dtype=float)

    loaded_edge_count = len(edge_load)
    loaded_tail_count = sum(1 for e in edge_load if e in tail_edges)

    tail_edge_load_mean = _safe_mean(tail_arr) if tail_arr.size else 0.0
    non_tail_edge_load_mean = _safe_mean(non_tail_arr) if non_tail_arr.size else 0.0

    tail_load_fraction = float(tail_total_load / total_load) if total_load > 0 else 0.0
    tail_edge_fraction_loaded = float(loaded_tail_count / loaded_edge_count) if loaded_edge_count > 0 else 0.0

    tail_efficiency_ratio = (
        tail_load_fraction / tail_edge_fraction_loaded
        if tail_edge_fraction_loaded > 0
        else 0.0
    )

    tail_burden_concentration = (
        tail_edge_load_mean / non_tail_edge_load_mean
        if non_tail_edge_load_mean > 0
        else 0.0
    )

    return {
        "tail_edge_load_mean": tail_edge_load_mean,
        "non_tail_edge_load_mean": non_tail_edge_load_mean,
        "tail_load_fraction": tail_load_fraction,
        "tail_edge_fraction_loaded": tail_edge_fraction_loaded,
        "tail_efficiency_ratio": tail_efficiency_ratio,
        "tail_burden_concentration": tail_burden_concentration,
    }

# ============================================================
# Maintenance cost
# ============================================================

def compute_maintenance_cost(
    G: nx.Graph,
    active_edge_penalty: float = 1.0,
    distal_weight_penalty_scale: float = 1.0,
    curvature_tail_penalty_scale: float = 0.0,
    edge_curvatures: dict[Edge, float] | None = None,
    tail_edges: set[Edge] | None = None,
) -> dict[str, float]:
    """
    Compute maintenance-style costs for activated distal infrastructure.

    Interpretation
    --------------
    This is a flexible phenomenological penalty model. It does not claim to be
    the uniquely correct physical maintenance law. It is a scaffold for testing
    whether sparse lower-tail structure can buy global transport savings at lower
    maintenance burden than more uniform distal activation.

    Cost components
    ---------------
    1. active_edge_penalty:
       Flat cost per active distal edge.

    2. distal_weight_penalty_scale:
       Additional penalty that grows as distal edges become "cheaper" / stronger.
       Since cheaper weight means stronger shortcut, we map lower edge weight to
       higher maintenance burden.

    3. curvature_tail_penalty_scale:
       Optional extra penalty on edges that sit in the lower curvature tail.
       Useful if you want the tail itself to be explicitly metabolically expensive.

    Returns
    -------
    dict
        {
            "n_active_distal_edges": float,
            "maintenance_cost": float,
            "tail_maintenance_cost": float,
            "mean_active_distal_weight": float,
        }
    """
    active_distal_edges = []
    active_weights = []

    for u, v, data in G.edges(data=True):
        if data.get("distal", False):
            edge = normalize_edge(u, v)
            active_distal_edges.append(edge)
            active_weights.append(float(data.get("weight", 1.0)))

    n_active = len(active_distal_edges)
    if n_active == 0:
        return {
            "n_active_distal_edges": 0.0,
            "maintenance_cost": 0.0,
            "tail_maintenance_cost": 0.0,
            "mean_active_distal_weight": 0.0,
        }

    active_weights_arr = np.asarray(active_weights, dtype=float)

    # Flat penalty per active distal edge
    flat_cost = active_edge_penalty * n_active

    # Lower weight = stronger/cheaper shortcut = more active infrastructure burden
    # We use inverse weight as a simple monotone proxy.
    weight_cost = distal_weight_penalty_scale * float(np.sum(1.0 / np.clip(active_weights_arr, 1e-8, None)))

    tail_cost = 0.0
    if curvature_tail_penalty_scale > 0.0 and edge_curvatures is not None and tail_edges is not None:
        for edge in active_distal_edges:
            if edge in tail_edges:
                tail_cost += curvature_tail_penalty_scale

    total_cost = flat_cost + weight_cost + tail_cost

    return {
        "n_active_distal_edges": float(n_active),
        "maintenance_cost": float(total_cost),
        "tail_maintenance_cost": float(tail_cost),
        "mean_active_distal_weight": _safe_mean(active_weights_arr),
    }


# ============================================================
# Objective function
# ============================================================

def compute_objective_J(
    transport_metrics: dict[str, Any],
    graph_metrics: dict[str, Any],
    maintenance_metrics: dict[str, Any],
    weights: dict[str, float],
) -> float:
    """
    Composite control-law objective.

    Default interpretation
    ----------------------
    J = transport term
      + maintenance term
      + congestion term
      + bottleneck / centralization term

    Expected keys in weights
    ------------------------
    transport
    maintenance
    congestion
    betweenness

    Returns
    -------
    float
    """
    w_transport = float(weights.get("transport", 1.0))
    w_maintenance = float(weights.get("maintenance", 1.0))
    w_congestion = float(weights.get("congestion", 0.0))
    w_betweenness = float(weights.get("betweenness", 0.0))

    transport_term = w_transport * float(transport_metrics.get("mean_transport_cost", 0.0))
    maintenance_term = w_maintenance * float(maintenance_metrics.get("maintenance_cost", 0.0))
    congestion_term = w_congestion * float(transport_metrics.get("max_edge_load", 0.0))
    betweenness_term = w_betweenness * float(graph_metrics.get("edge_betweenness_max", 0.0))

    return float(transport_term + maintenance_term + congestion_term + betweenness_term)


# ============================================================
# Optional diagnostics
# ============================================================

def build_path_edge_incidence_table(
    G: nx.Graph,
    n_pairs: int = 100,
    weighted: bool = True,
    rng: np.random.Generator | None = None,
    weight_attr: str = "weight",
) -> list[dict[str, Any]]:
    """
    Build a more detailed table of sampled source-target paths and their traversed edges.

    Useful for debugging or deeper downstream analyses.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nodes = np.asarray(list(G.nodes()))
    if len(nodes) < 2:
        return []

    rows: list[dict[str, Any]] = []
    path_weight = weight_attr if weighted else None

    for path_id in range(n_pairs):
        s, t = rng.choice(nodes, size=2, replace=False)
        s = int(s)
        t = int(t)

        path = nx.shortest_path(G, source=s, target=t, weight=path_weight)

        total_cost = 0.0
        for step_idx, (a, b) in enumerate(zip(path[:-1], path[1:])):
            edge = normalize_edge(a, b)
            edge_weight = float(G[a][b].get(weight_attr, 1.0)) if weighted else 1.0
            total_cost += edge_weight

            rows.append(
                {
                    "path_id": path_id,
                    "source": s,
                    "target": t,
                    "step_idx": step_idx,
                    "edge_u": edge[0],
                    "edge_v": edge[1],
                    "edge_weight": edge_weight,
                    "is_distal": bool(G[a][b].get("distal", False)),
                    "edge_class": G[a][b].get("edge_class", "unknown"),
                }
            )

        # annotate path total cost on final row for convenience
        if rows:
            rows[-1]["path_total_cost"] = total_cost

    return rows


def summarize_transport_regime(
    transport_metrics: dict[str, Any],
    graph_metrics: dict[str, Any],
    maintenance_metrics: dict[str, Any],
) -> dict[str, float]:
    """
    Convenience summary for quick inspection or logging.
    """
    return {
        "mean_transport_cost": float(transport_metrics.get("mean_transport_cost", float("nan"))),
        "max_edge_load": float(transport_metrics.get("max_edge_load", float("nan"))),
        "load_std": float(transport_metrics.get("load_std", float("nan"))),
        "mean_shortest_path": float(graph_metrics.get("mean_shortest_path", float("nan"))),
        "global_efficiency": float(graph_metrics.get("global_efficiency", float("nan"))),
        "edge_betweenness_max": float(graph_metrics.get("edge_betweenness_max", float("nan"))),
        "maintenance_cost": float(maintenance_metrics.get("maintenance_cost", float("nan"))),
    }