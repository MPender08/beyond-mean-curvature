from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RESULTS_DIR = Path("results/selective_tail")


def _base_config() -> dict[str, Any]:
    """
    Shared defaults across all selective-tail experiments.
    """
    return {
        "name": "base",
        "seed": 42,
        "rng": np.random.default_rng(42),

        # Graph structure
        "branching": 2,
        "depth": 4,
        "same_depth_only": True,
        "min_graph_distance": 4,
        "max_candidates": 150,

        # Distal edge weighting
        "max_distal_weight": 5.0,
        "min_distal_weight": 0.5,
        "default_distal_weight": 0.8,

        # Curvature settings
        "orc_alpha": 0.5,
        "weight_attr": "weight",
        "tail_quantile": 0.10,
        "tail_quantiles": (0.05, 0.10, 0.25),

        # Transport simulation
        "n_pairs": 1000,
        "weighted_transport": True,

        # Maintenance model
        "maintenance": {
            "active_edge_penalty": 1.0,
            "distal_weight_penalty_scale": 1.0,
            "curvature_tail_penalty_scale": 0.0,
        },

        # Composite objective
        "objective_weights": {
            "transport": 1.0,
            "maintenance": 1.0,
            "congestion": 0.03,
            "betweenness": 2.0,
        },

        # Output
        "results_dir": str(DEFAULT_RESULTS_DIR),

        # Condition list filled by presets
        "conditions": [],
    }


def _refresh_rng(config: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure RNG is consistent with config seed after any override.
    """
    config["rng"] = np.random.default_rng(config["seed"])
    return config


def _with_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    config["name"] = name
    return config


def _targeted_vs_random_config() -> dict[str, Any]:
    """
    First-pass falsification test:
    does sparse targeted distal placement outperform random sparse placement?
    """
    config = _base_config()
    _with_name(config, "targeted_vs_random")

    config["conditions"] = [
        {
            "name": "targeted_n05",
            "mode": "sparse_tail",
            "n_active": 5,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
        {
            "name": "random_n05",
            "mode": "sparse_tail",
            "n_active": 5,
            "distal_weight": 0.8,
            "placement_mode": "random",
        },
        {
            "name": "targeted_n10",
            "mode": "sparse_tail",
            "n_active": 10,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
        {
            "name": "random_n10",
            "mode": "sparse_tail",
            "n_active": 10,
            "distal_weight": 0.8,
            "placement_mode": "random",
        },
        {
            "name": "targeted_n20",
            "mode": "sparse_tail",
            "n_active": 20,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
        {
            "name": "random_n20",
            "mode": "sparse_tail",
            "n_active": 20,
            "distal_weight": 0.8,
            "placement_mode": "random",
        },
    ]

    return _refresh_rng(config)


def _betweenness_vs_curvature_config() -> dict[str, Any]:
    """
    Mechanism test:
    do the deepest negative-tail edges disproportionately carry transport?
    """
    config = _base_config()
    _with_name(config, "betweenness_vs_curvature")

    config["conditions"] = [
        {
            "name": "uniform_gamma_020",
            "mode": "uniform_gamma",
            "gamma": 0.20,
        },
        {
            "name": "uniform_gamma_040",
            "mode": "uniform_gamma",
            "gamma": 0.40,
        },
        {
            "name": "uniform_gamma_060",
            "mode": "uniform_gamma",
            "gamma": 0.60,
        },
        {
            "name": "uniform_gamma_080",
            "mode": "uniform_gamma",
            "gamma": 0.80,
        },
        {
            "name": "sparse_targeted_n10",
            "mode": "sparse_tail",
            "n_active": 10,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
        {
            "name": "sparse_targeted_n20",
            "mode": "sparse_tail",
            "n_active": 20,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
    ]

    return _refresh_rng(config)


def _matched_mean_vs_q10_config() -> dict[str, Any]:
    """
    Placeholder config for the most important ablation:
    matched mean curvature, different q10 depth.

    Note:
    This requires special graph-construction logic in build_graph_for_condition()
    that likely won't exist on day one. Keep as a forward-looking preset.
    """
    config = _base_config()
    _with_name(config, "matched_mean_vs_q10")

    # Slightly larger graph to make the tail structure more expressive.
    config["branching"] = 2
    config["depth"] = 5
    config["max_candidates"] = 250
    config["n_pairs"] = 1500

    config["conditions"] = [
        {
            "name": "uniform_meanmatch_candidate_A",
            "mode": "uniform_gamma",
            "gamma": 0.55,
        },
        {
            "name": "uniform_meanmatch_candidate_B",
            "mode": "uniform_gamma",
            "gamma": 0.70,
        },
        {
            "name": "tailmatch_sparse_n12",
            "mode": "sparse_tail",
            "n_active": 12,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
        {
            "name": "tailmatch_sparse_n24",
            "mode": "sparse_tail",
            "n_active": 24,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        },
    ]

    return _refresh_rng(config)


def _uniform_gamma_sweep_config() -> dict[str, Any]:
    """
    CAH_Control-adjacent baseline sweep.
    Useful for sanity checks before running more surgical tests.
    """
    config = _base_config()
    _with_name(config, "uniform_gamma_sweep")

    gammas = np.linspace(0.0, 1.0, 11)
    config["conditions"] = [
        {
            "name": f"uniform_gamma_{gamma:.2f}".replace(".", "p"),
            "mode": "uniform_gamma",
            "gamma": float(gamma),
        }
        for gamma in gammas
    ]

    return _refresh_rng(config)


def _targeted_scale_sweep_config() -> dict[str, Any]:
    """
    Sparse targeted scaling sweep.
    Tests how objective and tail structure evolve as more distal wormholes are opened.
    """
    config = _base_config()
    _with_name(config, "targeted_scale_sweep")

    config["conditions"] = [
        {
            "name": f"targeted_n{n_active:02d}",
            "mode": "sparse_tail",
            "n_active": n_active,
            "distal_weight": 0.8,
            "placement_mode": "targeted",
            "edge_score_fn": "max_depth_separation",
        }
        for n_active in [2, 4, 8, 12, 16, 24, 32]
    ]

    return _refresh_rng(config)


EXPERIMENT_BUILDERS = {
    "targeted_vs_random": _targeted_vs_random_config,
    "betweenness_vs_curvature": _betweenness_vs_curvature_config,
    "matched_mean_vs_q10": _matched_mean_vs_q10_config,
    "uniform_gamma_sweep": _uniform_gamma_sweep_config,
    "targeted_scale_sweep": _targeted_scale_sweep_config,
}


def get_experiment_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Return a fully instantiated experiment config.

    Parameters
    ----------
    name:
        Name of preset experiment.
    overrides:
        Optional shallow overrides on top of preset config.

    Returns
    -------
    dict
        Ready-to-run config dictionary.
    """
    if name not in EXPERIMENT_BUILDERS:
        valid = ", ".join(sorted(EXPERIMENT_BUILDERS))
        raise ValueError(f"Unknown experiment config: {name}. Valid options: {valid}")

    config = deepcopy(EXPERIMENT_BUILDERS[name]())

    if overrides:
        for key, value in overrides.items():
            config[key] = value

    return _refresh_rng(config)