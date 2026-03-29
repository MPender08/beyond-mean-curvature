from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import csv
import traceback

from experiment_configs import get_experiment_config
from selective_tail_utils import (
    build_base_graph,
    generate_distal_candidates,
    annotate_edge_classes,
    build_uniform_gamma_graph,
    build_sparse_tail_graph,
    build_rewired_tail_graph,
)
from curvature_metrics import (
    compute_ollivier_ricci_curvatures,
    summarize_curvature_distribution,
    get_tail_edges,
)
from transport_metrics import (
    simulate_transport,
    compute_graph_metrics,
    compute_tail_load_metrics,
    compute_maintenance_cost,
    compute_objective_J,
)
from plotting_selective_tail import make_all_plots




# ============================================================
# Paths / saving
# ============================================================

DEFAULT_RESULTS_DIR = Path("results/selective_tail")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_results(payload: dict[str, Any], config: dict[str, Any]) -> None:
    """
    Save result tables and metadata for a run.
    """
    base_dir = Path(config.get("results_dir", DEFAULT_RESULTS_DIR))
    csv_dir = base_dir / "csv"
    json_dir = base_dir / "json"

    ensure_dir(csv_dir)
    ensure_dir(json_dir)

    experiment_name = config["name"]
    results = payload["results"]

    if results:
        csv_path = csv_dir / f"{experiment_name}.csv"

        fieldnames = sorted({key for row in results for key in row.keys()})

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(results)

    json_path = json_dir / f"{experiment_name}_metadata.json"
    
	
    config_for_json = dict(config)
    if "rng" in config_for_json:
        config_for_json["rng"] = f"np.random.default_rng({config['seed']})"
    
    metadata = {
        "config": config_for_json,
        "n_conditions": len(results),
        "condition_names": [r["condition_name"] for r in results],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    edge_json_path = json_dir / f"{experiment_name}_edge_payloads.json"
    serializable_edge_payloads = serialize_edge_payloads(payload["edge_level_payloads"])
    with edge_json_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_edge_payloads, f, indent=2)


def serialize_edge_payloads(edge_level_payloads: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Convert sets / Counters / tuple keys into JSON-safe structures.
    """
    out: list[dict[str, Any]] = []

    for condition_name, payload in edge_level_payloads:
        edge_curvatures = payload.get("edge_curvatures", {})
        edge_load = payload.get("edge_load", {})
        tail_edges = payload.get("tail_edges", set())

        out.append(
            {
                "condition_name": condition_name,
                "edge_curvatures": [
                    {"edge": list(edge), "curvature": float(curv)}
                    for edge, curv in edge_curvatures.items()
                ],
                "edge_load": [
                    {"edge": list(edge), "load": int(load)}
                    for edge, load in edge_load.items()
                ],
                "tail_edges": [list(edge) for edge in sorted(tail_edges)],
            }
        )

    return out


# ============================================================
# Condition building
# ============================================================

def build_graph_for_condition(
    base_G,
    distal_candidates,
    condition: dict[str, Any],
    config: dict[str, Any],
):
    """
    Dispatch graph construction based on condition mode.
    """
    mode = condition["mode"]

    if mode == "uniform_gamma":
        return build_uniform_gamma_graph(
            base_G=base_G,
            distal_candidates=distal_candidates,
            gamma=condition["gamma"],
            max_distal_weight=condition.get(
                "max_distal_weight",
                config.get("max_distal_weight", 5.0),
            ),
            min_distal_weight=condition.get(
                "min_distal_weight",
                config.get("min_distal_weight", 0.5),
            ),
        )

    if mode == "sparse_tail":
        return build_sparse_tail_graph(
            base_G=base_G,
            distal_candidates=distal_candidates,
            n_active=condition["n_active"],
            distal_weight=condition.get("distal_weight", config.get("default_distal_weight", 0.8)),
            placement_mode=condition.get("placement_mode", "random"),
            edge_score_fn=condition.get("edge_score_fn"),
            rng=config["rng"],
        )

    if mode == "rewired_tail_control":
        if "active_distal_edges" not in condition:
            raise ValueError(
                "rewired_tail_control requires 'active_distal_edges' in condition."
            )
        return build_rewired_tail_graph(
            base_G=base_G,
            active_distal_edges=condition["active_distal_edges"],
            distal_candidates=distal_candidates,
            distal_weight=condition.get("distal_weight", config.get("default_distal_weight", 0.8)),
            rng=config["rng"],
        )

    raise ValueError(f"Unknown condition mode: {mode}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_condition(
    base_G,
    distal_candidates,
    condition: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build graph, compute all metrics, and return:
        row: flat condition summary
        payload: edge-level diagnostic data
    """
    G = build_graph_for_condition(base_G, distal_candidates, condition, config)
    G = annotate_edge_classes(G)

    edge_curvatures = compute_ollivier_ricci_curvatures(
        G,
        alpha=config.get("orc_alpha", 0.5),
        weight_attr=config.get("weight_attr", "weight"),
	allow_fallback=False,
    )

    curvature_summary = summarize_curvature_distribution(
        edge_curvatures,
        tail_quantiles=config.get("tail_quantiles", (0.05, 0.10, 0.25)),
    )

    tail_quantile = config.get("tail_quantile", 0.10)
    tail_edges = get_tail_edges(edge_curvatures, quantile=tail_quantile)

    transport = simulate_transport(
        G,
        n_pairs=config.get("n_pairs", 1000),
        weighted=config.get("weighted_transport", True),
        rng=config["rng"],
    )

    graph_metrics = compute_graph_metrics(G)

    tail_load_metrics = compute_tail_load_metrics(
        edge_load=transport["edge_load"],
        tail_edges=tail_edges,
    )

    maintenance = compute_maintenance_cost(
        G,
        active_edge_penalty=config["maintenance"].get("active_edge_penalty", 1.0),
        distal_weight_penalty_scale=config["maintenance"].get("distal_weight_penalty_scale", 1.0),
        curvature_tail_penalty_scale=config["maintenance"].get("curvature_tail_penalty_scale", 0.0),
        edge_curvatures=edge_curvatures,
        tail_edges=tail_edges,
    )

    objective_J = compute_objective_J(
        transport_metrics=transport,
        graph_metrics=graph_metrics,
        maintenance_metrics=maintenance,
        weights=config["objective_weights"],
    )

    row: dict[str, Any] = {
        "experiment_name": config["name"],
        "condition_name": condition["name"],
        "intervention_mode": condition["mode"],
        "seed": config["seed"],
        "branching": config["branching"],
        "depth": config["depth"],
        "n_pairs": config.get("n_pairs", 1000),
        "tail_quantile": tail_quantile,
        "objective_J": float(objective_J),
    }

    # Condition parameters
    if "gamma" in condition:
        row["gamma"] = condition["gamma"]
    if "n_active" in condition:
        row["n_active"] = condition["n_active"]
    if "distal_weight" in condition:
        row["distal_weight"] = condition["distal_weight"]
    if "placement_mode" in condition:
        row["placement_mode"] = condition["placement_mode"]
    if "edge_score_fn" in condition:
        row["edge_score_fn"] = condition["edge_score_fn"]

    # Flatten summaries
    row.update(curvature_summary)

    # transport includes edge_load and pair_costs; omit large objects
    for key, value in transport.items():
        if key not in {"edge_load", "pair_costs"}:
            row[key] = value

    row.update(graph_metrics)
    row.update(tail_load_metrics)
    row.update(maintenance)

    payload = {
        "edge_curvatures": edge_curvatures,
        "edge_load": transport["edge_load"],
        "tail_edges": tail_edges,
        "pair_costs": transport.get("pair_costs", []),
    }

    return row, payload


# ============================================================
# Driver
# ============================================================

def run_selective_tail_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """
    Main experiment runner.

    Returns:
        {
            "results": [flat row dicts...],
            "edge_level_payloads": [(condition_name, payload), ...],
        }
    """
    base_G = build_base_graph(
        branching=config["branching"],
        depth=config["depth"],
    )

    distal_candidates = generate_distal_candidates(
        base_G,
        same_depth_only=config.get("same_depth_only", True),
        min_graph_distance=config.get("min_graph_distance", 4),
        max_candidates=config.get("max_candidates", 200),
        rng=config["rng"],
    )

    results: list[dict[str, Any]] = []
    edge_level_payloads: list[tuple[str, dict[str, Any]]] = []

    print("=" * 72)
    print(f"Running selective tail experiment: {config['name']}")
    print(f"branching={config['branching']} depth={config['depth']} seed={config['seed']}")
    print(f"n_conditions={len(config['conditions'])}")
    print(f"n_distal_candidates={len(distal_candidates)}")
    print("=" * 72)

    for idx, condition in enumerate(config["conditions"], start=1):
        print(f"[{idx}/{len(config['conditions'])}] Evaluating condition: {condition['name']}")

        try:
            row, payload = evaluate_condition(
                base_G=base_G,
                distal_candidates=distal_candidates,
                condition=condition,
                config=config,
            )
            results.append(row)
            edge_level_payloads.append((condition["name"], payload))

            print(
                f"  -> J={row['objective_J']:.4f} "
                f"mean_curv={row.get('mean_curvature', float('nan')):.4f} "
                f"q10={row.get('q10_curvature', float('nan')):.4f} "
                f"transport={row.get('mean_transport_cost', float('nan')):.4f}"
            )
        except Exception as exc:
            print(f"  !! Condition failed: {condition['name']}")
            print(f"  !! {exc}")
            traceback.print_exc()

    return {
        "results": results,
        "edge_level_payloads": edge_level_payloads,
    }

# =============================================================
# Repeater
#==============================================================

def run_repeated_config(config_name: str, seeds: list[int]) -> list[dict]:
    all_payloads = []
    for seed in seeds:
        config = get_experiment_config(
            config_name,
            overrides={
                "seed": seed,
                "results_dir": f"results/selective_tail/seed_{seed}",
            },
        )
        payload = run_selective_tail_experiment(config)
        save_results(payload, config)
        make_all_plots(payload, config)
        all_payloads.append({
            "seed": seed,
            "results": payload["results"],
        })
    return all_payloads

# ============================================================
# Entry point
# ============================================================

def main(config_name: str = "targeted_vs_random") -> dict[str, Any]:
    config = get_experiment_config(config_name)

    # Attach a reusable RNG if your config factory has not already done so.
    if "rng" not in config:
        import numpy as np
        config["rng"] = np.random.default_rng(config["seed"])

    payload = run_selective_tail_experiment(config)
    save_results(payload, config)

    try:
        make_all_plots(payload, config)
    except Exception as exc:
        print("Plotting failed.")
        print(exc)

    print("Done.")
    return payload


if __name__ == "__main__":
    seeds = [42, 43, 44, 45, 46]
    all_payloads = run_repeated_config("targeted_vs_random", seeds)