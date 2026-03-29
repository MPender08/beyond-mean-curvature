from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np


Edge = tuple[int, int]
ScoreFn = Callable[[nx.Graph, list[Edge]], dict[Edge, float]]


# ============================================================
# Base graph construction
# ============================================================

def build_base_graph(branching: int, depth: int) -> nx.Graph:
    """
    Build a balanced tree as the base hierarchical substrate.

    Parameters
    ----------
    branching:
        Number of children per non-leaf node.
    depth:
        Tree depth (networkx balanced_tree h parameter).

    Returns
    -------
    nx.Graph
        Undirected weighted graph with basal edges initialized to weight=1.0.
    """
    G = nx.balanced_tree(r=branching, h=depth, create_using=nx.Graph)

    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distal"] = False
        G[u][v]["edge_class"] = "basal"

    return G


def node_depths(G: nx.Graph, root: int = 0) -> dict[int, int]:
    """
    Compute graph-theoretic depth of each node from the root.
    """
    return dict(nx.single_source_shortest_path_length(G, root))


# ============================================================
# Distal candidate generation
# ============================================================

def generate_distal_candidates(
    G: nx.Graph,
    same_depth_only: bool = True,
    min_graph_distance: int = 4,
    max_candidates: int = 200,
    rng: np.random.Generator | None = None,
    root: int = 0,
) -> list[Edge]:
    """
    Generate candidate distal shortcut edges.

    Strategy
    --------
    Prefer non-adjacent node pairs that are:
    - not already connected
    - reasonably far apart in the base graph
    - optionally at the same depth, to mimic branch-bridging shortcuts

    Parameters
    ----------
    G:
        Base hierarchy.
    same_depth_only:
        Restrict candidates to nodes at the same depth.
    min_graph_distance:
        Minimum shortest-path distance in the base graph.
    max_candidates:
        Maximum number of candidates returned.
    rng:
        Random generator for candidate shuffling.
    root:
        Root node for depth calculation.

    Returns
    -------
    list[Edge]
        Sorted edge tuples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    depths = node_depths(G, root=root)
    nodes = list(G.nodes())
    candidates: list[Edge] = []

    for u, v in combinations(nodes, 2):
        if G.has_edge(u, v):
            continue

        if same_depth_only and depths[u] != depths[v]:
            continue

        d = nx.shortest_path_length(G, u, v)
        if d < min_graph_distance:
            continue

        candidates.append(tuple(sorted((u, v))))

    rng.shuffle(candidates)
    return candidates[:max_candidates]


# ============================================================
# Edge scoring functions for targeted placement
# ============================================================

def score_edges_by_depth_separation(
    G: nx.Graph,
    candidate_edges: list[Edge],
    root: int = 0,
) -> dict[Edge, float]:
    """
    Score candidates by graph distance, with a small bonus for deep nodes.

    This is a simple heuristic for 'targeted' wormhole placement.
    """
    depths = node_depths(G, root=root)
    scores: dict[Edge, float] = {}

    for u, v in candidate_edges:
        path_d = nx.shortest_path_length(G, u, v)
        depth_bonus = 0.5 * (depths[u] + depths[v])
        scores[(u, v)] = float(path_d + depth_bonus)

    return scores


def score_edges_by_branch_separation(
    G: nx.Graph,
    candidate_edges: list[Edge],
    root: int = 0,
) -> dict[Edge, float]:
    """
    Approximate branch separation using lowest common ancestor depth proxy.

    Higher score means nodes likely reside in more distinct branches.
    """
    depths = node_depths(G, root=root)
    root_paths = nx.single_source_shortest_path(G, root)
    scores: dict[Edge, float] = {}

    for u, v in candidate_edges:
        path_u = root_paths[u]
        path_v = root_paths[v]

        common_depth = 0
        for a, b in zip(path_u, path_v):
            if a == b:
                common_depth += 1
            else:
                break

        # Smaller shared prefix means more branch-separated.
        shared_prefix_penalty = common_depth
        path_d = nx.shortest_path_length(G, u, v)
        depth_bonus = 0.25 * (depths[u] + depths[v])

        scores[(u, v)] = float(path_d + depth_bonus - shared_prefix_penalty)

    return scores


def resolve_edge_score_fn(
    edge_score_fn: str | ScoreFn | None,
) -> ScoreFn:
    """
    Resolve string or callable edge score function.
    """
    if edge_score_fn is None:
        return score_edges_by_depth_separation

    if callable(edge_score_fn):
        return edge_score_fn

    if edge_score_fn == "max_depth_separation":
        return score_edges_by_depth_separation

    if edge_score_fn == "max_branch_separation":
        return score_edges_by_branch_separation

    raise ValueError(f"Unknown edge_score_fn: {edge_score_fn}")


# ============================================================
# Graph annotation helpers
# ============================================================

def annotate_edge_classes(G: nx.Graph, root: int = 0) -> nx.Graph:
    """
    Add consistent metadata to each edge.

    Added / normalized attributes:
    - weight
    - distal
    - edge_class
    - mean_node_depth
    - graph_distance_if_distal (for distal edges, if present)
    """
    depths = node_depths(G, root=root)

    for u, v in G.edges():
        data = G[u][v]

        data.setdefault("weight", 1.0)
        data.setdefault("distal", False)

        if data["distal"]:
            data["edge_class"] = "distal"
        else:
            data["edge_class"] = "basal"

        data["mean_node_depth"] = 0.5 * (depths[u] + depths[v])

        if data["distal"]:
            # If edge was not in the original tree, the current shortest-path is now 1,
            # so preserve its original separation if available or leave missing.
            data.setdefault("graph_distance_if_distal", None)

    return G


# ============================================================
# Core graph-building interventions
# ============================================================

def build_uniform_gamma_graph(
    base_G: nx.Graph,
    distal_candidates: list[Edge],
    gamma: float,
    max_distal_weight: float = 5.0,
    min_distal_weight: float = 0.5,
) -> nx.Graph:
    """
    CAH_Control-style intervention:
    as gamma increases,
    - more distal edges become active
    - active distal edges become cheaper

    Parameters
    ----------
    gamma:
        Scalar in [0, 1].
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    G = base_G.copy()

    distal_weight = max_distal_weight - gamma * (max_distal_weight - min_distal_weight)
    n_active = int(round(gamma * len(distal_candidates)))
    active_edges = distal_candidates[:n_active]

    for u, v in active_edges:
        G.add_edge(
            u,
            v,
            weight=float(distal_weight),
            distal=True,
            edge_class="distal",
            graph_distance_if_distal=nx.shortest_path_length(base_G, u, v),
        )

    # Normalize basal edge attributes
    for u, v in G.edges():
        if "distal" not in G[u][v]:
            G[u][v]["distal"] = False
            G[u][v]["weight"] = 1.0
            G[u][v]["edge_class"] = "basal"

    return G


def build_sparse_tail_graph(
    base_G: nx.Graph,
    distal_candidates: list[Edge],
    n_active: int,
    distal_weight: float,
    placement_mode: str = "targeted",
    edge_score_fn: str | ScoreFn | None = None,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    Activate a sparse subset of distal candidates.

    Parameters
    ----------
    n_active:
        Number of distal edges to activate.
    distal_weight:
        Weight assigned to all active distal edges.
    placement_mode:
        "random" or "targeted"
    edge_score_fn:
        Scoring rule used when placement_mode == "targeted"
    rng:
        Random generator

    Returns
    -------
    nx.Graph
    """
    if rng is None:
        rng = np.random.default_rng(42)

    G = base_G.copy()

    if n_active < 0:
        raise ValueError(f"n_active must be >= 0, got {n_active}")

    n_active = min(n_active, len(distal_candidates))

    if placement_mode == "random":
        candidate_pool = distal_candidates.copy()
        rng.shuffle(candidate_pool)
        active_edges = candidate_pool[:n_active]

    elif placement_mode == "targeted":
        score_fn = resolve_edge_score_fn(edge_score_fn)
        scores = score_fn(base_G, distal_candidates)

        # Highest scores first
        active_edges = sorted(
            distal_candidates,
            key=lambda e: scores[e],
            reverse=True,
        )[:n_active]

    else:
        raise ValueError(f"Unknown placement_mode: {placement_mode}")

    for u, v in active_edges:
        G.add_edge(
            u,
            v,
            weight=float(distal_weight),
            distal=True,
            edge_class="distal",
            graph_distance_if_distal=nx.shortest_path_length(base_G, u, v),
        )

    for u, v in G.edges():
        if "distal" not in G[u][v]:
            G[u][v]["distal"] = False
            G[u][v]["weight"] = 1.0
            G[u][v]["edge_class"] = "basal"

    return G


def build_rewired_tail_graph(
    base_G: nx.Graph,
    active_distal_edges: list[Edge],
    distal_candidates: list[Edge],
    distal_weight: float,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    Preserve the number of active distal edges but randomize their placement.

    This is intended as a control for 'targeted routing skeleton' hypotheses.

    Parameters
    ----------
    active_distal_edges:
        The originally selected active distal set. Only its length is preserved here.
    distal_candidates:
        Pool from which to randomly redraw.
    distal_weight:
        Weight assigned to all rewired distal edges.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_active = len(active_distal_edges)
    candidate_pool = distal_candidates.copy()
    rng.shuffle(candidate_pool)
    rewired_edges = candidate_pool[:n_active]

    G = base_G.copy()

    for u, v in rewired_edges:
        G.add_edge(
            u,
            v,
            weight=float(distal_weight),
            distal=True,
            edge_class="distal",
            graph_distance_if_distal=nx.shortest_path_length(base_G, u, v),
        )

    for u, v in G.edges():
        if "distal" not in G[u][v]:
            G[u][v]["distal"] = False
            G[u][v]["weight"] = 1.0
            G[u][v]["edge_class"] = "basal"

    return G


# ============================================================
# Optional utilities for future experiments
# ============================================================

def get_active_distal_edges(G: nx.Graph) -> list[Edge]:
    """
    Return sorted active distal edges from a graph.
    """
    edges: list[Edge] = []
    for u, v, data in G.edges(data=True):
        if data.get("distal", False):
            edges.append(tuple(sorted((u, v))))
    return sorted(edges)


def count_active_distal_edges(G: nx.Graph) -> int:
    """
    Convenience counter.
    """
    return len(get_active_distal_edges(G))


def summarize_distal_edge_lengths(G: nx.Graph) -> dict[str, float]:
    """
    Summarize original graph separation of distal edges, if annotated.
    """
    lengths = []
    for _, _, data in G.edges(data=True):
        if data.get("distal", False) and data.get("graph_distance_if_distal") is not None:
            lengths.append(float(data["graph_distance_if_distal"]))

    if not lengths:
        return {
            "distal_length_mean": 0.0,
            "distal_length_median": 0.0,
            "distal_length_max": 0.0,
        }

    arr = np.asarray(lengths, dtype=float)
    return {
        "distal_length_mean": float(np.mean(arr)),
        "distal_length_median": float(np.median(arr)),
        "distal_length_max": float(np.max(arr)),
    }