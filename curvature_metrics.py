from __future__ import annotations

from typing import Any

import math
import numpy as np
import networkx as nx

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    _HAS_ORC = True
except Exception:
    OllivierRicci = None
    _HAS_ORC = False


Edge = tuple[int, int]


# ============================================================
# Helpers
# ============================================================

def normalize_edge(u: int, v: int) -> Edge:
    """
    Store edges in canonical sorted order.
    """
    return (u, v) if u <= v else (v, u)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


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


def clear_orc_internal_caches() -> None:
    """
    Clear GraphRicciCurvature module-level lru_cache'd functions, if present.

    This protects repeated experiments where multiple distinct graphs reuse
    the same node IDs, which can otherwise lead to stale neighborhood-mass
    cache reuse across runs.
    """
    try:
        import GraphRicciCurvature.OllivierRicci as OR
    except Exception:
        return

    for attr_name in dir(OR):
        attr = getattr(OR, attr_name)
        if callable(attr) and hasattr(attr, "cache_clear"):
            try:
                attr.cache_clear()
            except Exception:
                pass


# ============================================================
# Core curvature computation
# ============================================================

import sys
import multiprocessing as mp

if sys.platform == "win32":
    class DummyPool:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def imap_unordered(self, func, iterable, chunksize=1):
            return map(func, iterable)
        def map(self, func, iterable, chunksize=None):
            return list(map(func, iterable))
        def close(self):
            pass
        def join(self):
            pass

    class DummyContext:
        def Pool(self, *args, **kwargs):
            return DummyPool()

    mp.get_context = lambda method=None: DummyContext()


def compute_ollivier_ricci_curvatures(
    G: nx.Graph,
    alpha: float = 0.5,
    weight_attr: str = "weight",
    method: str = "OTD",
    proc: int = 1,
    verbose: str = "ERROR",
    allow_fallback: bool = False,
) -> dict[Edge, float]:
    """
    Compute Ollivier-Ricci curvature on every edge of the graph.

    Parameters
    ----------
    G:
        Input graph.
    alpha:
        ORC alpha parameter.
    weight_attr:
        Edge attribute used as weight/distance.
    method:
        Passed through to GraphRicciCurvature when available.
    proc:
        Number of worker processes for ORC implementation.
    verbose:
        Logging mode for GraphRicciCurvature.
    allow_fallback:
        If True and GraphRicciCurvature is unavailable, return a simple
        surrogate edge score instead of raising an error.

    Returns
    -------
    dict[Edge, float]
        Mapping from normalized edge -> ricci curvature.

    Notes
    -----
    - Preferred behavior is to use GraphRicciCurvature directly.
    - Fallback mode is only for development sanity checks and should not
      be treated as true ORC.
    """
    if _HAS_ORC:
        clear_orc_internal_caches()
        
        H = G.copy()

        # GraphRicciCurvature reads "weight" by default, so if the user wants a
        # different attribute, copy it over into "weight".
        if weight_attr != "weight":
            for u, v in H.edges():
                if weight_attr not in H[u][v]:
                    raise KeyError(f"Edge {(u, v)} missing weight attribute '{weight_attr}'")
                H[u][v]["weight"] = H[u][v][weight_attr]

        orc = OllivierRicci(
            H,
            alpha=alpha,
            method=method,
            proc=proc,
            verbose=verbose,
        )
        orc.compute_ricci_curvature()

        edge_curvatures: dict[Edge, float] = {}
        for u, v, data in orc.G.edges(data=True):
            curv = data.get("ricciCurvature", None)
            if curv is None:
                raise RuntimeError(f"ricciCurvature missing on edge {(u, v)} after ORC computation")
            edge_curvatures[normalize_edge(u, v)] = float(curv)

        return edge_curvatures

    if allow_fallback:
        return compute_surrogate_curvatures(G, weight_attr=weight_attr)

    raise ImportError(
        "GraphRicciCurvature is not available and allow_fallback=False. "
        "Install GraphRicciCurvature or enable fallback for debugging only."
    )


def compute_surrogate_curvatures(
    G: nx.Graph,
    weight_attr: str = "weight",
) -> dict[Edge, float]:
    """
    Development-only surrogate for curvature-like edge ranking.

    This is NOT Ollivier-Ricci curvature.

    The goal is simply to provide a monotonic proxy that:
    - penalizes long/heavy edges less
    - rewards edges that connect nodes with lower weighted shortest-path
      alternative routes poorly (i.e. edges acting as strong shortcuts)

    One heuristic used here:
        surrogate = alt_path_cost_without_edge - direct_edge_weight

    Then mapped through a signed squashing transform so magnitudes remain stable.

    Interpretation:
    - Larger positive surrogate raw values mean the edge is a more important shortcut.
    - After sign inversion / squashing, more "shortcut-like" edges become more negative,
      to roughly mimic the lower-tail semantics of ORC.

    Returns
    -------
    dict[Edge, float]
    """
    H = G.copy()
    edge_scores: dict[Edge, float] = {}

    for u, v in list(H.edges()):
        edge = normalize_edge(u, v)
        w = float(H[u][v].get(weight_attr, 1.0))

        H.remove_edge(u, v)
        try:
            alt_cost = nx.shortest_path_length(H, u, v, weight=weight_attr)
            raw = alt_cost - w
        except nx.NetworkXNoPath:
            raw = 10.0  # large shortcut importance if edge removal disconnects graph
        finally:
            H.add_edge(u, v, **G[u][v])

        # Map to pseudo-curvature: more shortcut-like => more negative
        pseudo_curv = -math.tanh(raw / 5.0)
        edge_scores[edge] = float(pseudo_curv)

    return edge_scores

# ============================================================
# Compute Tail Load Metrics
# ============================================================

def compute_tail_load_metrics(
    edge_load: dict[Edge, int] | Counter[Edge],
    tail_edges: set[Edge],
) -> dict[str, float]:
    """
    Compare traffic carried by lower-tail curvature edges vs the rest.

    Returns
    -------
    dict
        {
            "tail_edge_load_mean": float,
            "non_tail_edge_load_mean": float,
            "tail_load_fraction": float,
            "tail_edge_fraction_loaded": float,
            "tail_efficiency_ratio": float,
        }
    """
    if not edge_load:
        return {
            "tail_edge_load_mean": 0.0,
            "non_tail_edge_load_mean": 0.0,
            "tail_load_fraction": 0.0,
            "tail_edge_fraction_loaded": 0.0,
            "tail_efficiency_ratio": 0.0,
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

    tail_load_fraction = float(tail_total_load / total_load) if total_load > 0 else 0.0
    tail_edge_fraction_loaded = float(loaded_tail_count / loaded_edge_count) if loaded_edge_count > 0 else 0.0

    tail_efficiency_ratio = (
        tail_load_fraction / tail_edge_fraction_loaded
        if tail_edge_fraction_loaded > 0
        else 0.0
    )

    return {
        "tail_edge_load_mean": _safe_mean(tail_arr) if tail_arr.size else 0.0,
        "non_tail_edge_load_mean": _safe_mean(non_tail_arr) if non_tail_arr.size else 0.0,
        "tail_load_fraction": tail_load_fraction,
        "tail_edge_fraction_loaded": tail_edge_fraction_loaded,
        "tail_efficiency_ratio": tail_efficiency_ratio,
    }

# ============================================================
# Distribution summaries
# ============================================================

def summarize_curvature_distribution(
    edge_curvatures: dict[Edge, float],
    tail_quantiles: tuple[float, ...] = (0.05, 0.10, 0.25),
) -> dict[str, float]:
    """
    Summarize the edge curvature distribution.

    Parameters
    ----------
    edge_curvatures:
        Mapping edge -> curvature
    tail_quantiles:
        Quantiles to summarize, typically lower-tail cutoffs.

    Returns
    -------
    dict[str, float]
    """
    vals = np.asarray(list(edge_curvatures.values()), dtype=float)

    out: dict[str, float] = {
        "n_edges_curvature": float(vals.size),
        "mean_curvature": _safe_mean(vals),
        "median_curvature": _safe_median(vals),
        "min_curvature": float(np.min(vals)) if vals.size else float("nan"),
        "max_curvature": float(np.max(vals)) if vals.size else float("nan"),
        "curvature_std": _safe_std(vals),
    }

    for q in tail_quantiles:
        q_pct = int(round(100 * q))
        out[f"q{q_pct:02d}_curvature"] = _safe_quantile(vals, q)

    return out


def get_tail_edges(
    edge_curvatures: dict[Edge, float],
    quantile: float = 0.10,
) -> set[Edge]:
    """
    Return the set of edges in the lower curvature tail.

    Parameters
    ----------
    edge_curvatures:
        Mapping edge -> curvature
    quantile:
        Lower-tail cutoff. Example: 0.10 means q10 tail.

    Returns
    -------
    set[Edge]
    """
    if not edge_curvatures:
        return set()

    vals = np.asarray(list(edge_curvatures.values()), dtype=float)
    threshold = float(np.quantile(vals, quantile))

    return {
        edge for edge, curv in edge_curvatures.items()
        if curv <= threshold
    }


def get_non_tail_edges(
    edge_curvatures: dict[Edge, float],
    quantile: float = 0.10,
) -> set[Edge]:
    """
    Convenience complement of get_tail_edges().
    """
    tail_edges = get_tail_edges(edge_curvatures, quantile=quantile)
    return set(edge_curvatures.keys()) - tail_edges


# ============================================================
# Tail-vs-bulk diagnostics
# ============================================================

def compare_mean_vs_tail(
    edge_curvatures: dict[Edge, float],
    quantile: float = 0.10,
) -> dict[str, float]:
    """
    Compare lower-tail curvature to the rest of the distribution.

    Useful for sanity checks when asking whether q10 contains information
    not captured by the mean.

    Returns
    -------
    dict[str, float]
    """
    if not edge_curvatures:
        return {
            "tail_mean_curvature": float("nan"),
            "non_tail_mean_curvature": float("nan"),
            "tail_minus_non_tail_curvature": float("nan"),
            "tail_fraction": float("nan"),
        }

    tail_edges = get_tail_edges(edge_curvatures, quantile=quantile)

    tail_vals = np.asarray(
        [edge_curvatures[e] for e in tail_edges],
        dtype=float,
    )
    non_tail_vals = np.asarray(
        [curv for e, curv in edge_curvatures.items() if e not in tail_edges],
        dtype=float,
    )

    tail_mean = _safe_mean(tail_vals)
    non_tail_mean = _safe_mean(non_tail_vals)

    return {
        "tail_mean_curvature": tail_mean,
        "non_tail_mean_curvature": non_tail_mean,
        "tail_minus_non_tail_curvature": (
            tail_mean - non_tail_mean
            if not (math.isnan(tail_mean) or math.isnan(non_tail_mean))
            else float("nan")
        ),
        "tail_fraction": float(len(tail_edges) / max(len(edge_curvatures), 1)),
    }


# ============================================================
# Edge-level joins / diagnostics
# ============================================================

def build_edge_metric_table(
    G: nx.Graph,
    edge_curvatures: dict[Edge, float],
    edge_load: dict[Edge, int] | None = None,
    quantile: float = 0.10,
) -> list[dict[str, Any]]:
    """
    Join per-edge curvature with graph metadata and optional load.

    Returns a list of row dicts suitable for later plotting or CSV export.
    """
    tail_edges = get_tail_edges(edge_curvatures, quantile=quantile)
    rows: list[dict[str, Any]] = []

    for u, v, data in G.edges(data=True):
        edge = normalize_edge(u, v)
        rows.append(
            {
                "edge_u": edge[0],
                "edge_v": edge[1],
                "curvature": float(edge_curvatures.get(edge, float("nan"))),
                "is_tail_edge": edge in tail_edges,
                "weight": float(data.get("weight", 1.0)),
                "distal": bool(data.get("distal", False)),
                "edge_class": data.get("edge_class", "unknown"),
                "mean_node_depth": float(data.get("mean_node_depth", float("nan"))),
                "graph_distance_if_distal": (
                    float(data["graph_distance_if_distal"])
                    if data.get("graph_distance_if_distal") is not None
                    else float("nan")
                ),
                "edge_load": int(edge_load.get(edge, 0)) if edge_load is not None else 0,
            }
        )

    return rows


def compute_curvature_load_correlation(
    edge_curvatures: dict[Edge, float],
    edge_load: dict[Edge, int],
) -> dict[str, float]:
    """
    Compute simple correlation diagnostics between curvature and transport load.

    Notes
    -----
    Negative correlation would be consistent with lower-curvature edges
    carrying higher load.
    """
    shared_edges = [e for e in edge_curvatures if e in edge_load]
    if len(shared_edges) < 2:
        return {
            "corr_curvature_vs_load": float("nan"),
            "corr_abs_curvature_vs_load": float("nan"),
        }

    curv = np.asarray([edge_curvatures[e] for e in shared_edges], dtype=float)
    load = np.asarray([edge_load[e] for e in shared_edges], dtype=float)

    corr_signed = np.corrcoef(curv, load)[0, 1]
    corr_abs = np.corrcoef(np.abs(curv), load)[0, 1]

    return {
        "corr_curvature_vs_load": float(corr_signed),
        "corr_abs_curvature_vs_load": float(corr_abs),
    }


# ============================================================
# Optional matching helpers for future experiments
# ============================================================

def curvature_distance(
    summary_a: dict[str, float],
    summary_b: dict[str, float],
    key: str = "mean_curvature",
) -> float:
    """
    Simple absolute difference helper for matching conditions.
    """
    a = summary_a.get(key, float("nan"))
    b = summary_b.get(key, float("nan"))
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    return float(abs(a - b))


def rank_condition_match_candidates(
    summaries: list[dict[str, Any]],
    target_value: float,
    key: str,
) -> list[dict[str, Any]]:
    """
    Sort summary rows by proximity to a target metric value.
    """
    def _dist(row: dict[str, Any]) -> float:
        value = row.get(key, float("nan"))
        if math.isnan(value):
            return float("inf")
        return abs(value - target_value)

    return sorted(summaries, key=_dist)