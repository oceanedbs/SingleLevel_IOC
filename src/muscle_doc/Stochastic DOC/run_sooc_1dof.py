#!/usr/bin/env python3
"""Run the Step-2 one-joint stochastic optimal open-loop control example."""

from __future__ import annotations

import numpy as np

from sooc_1dof_func import (
    default_sooc_parameters,
    instantiate_sooc_1dof_model,
    make_sooc_1dof_model,
    numerize_var,
    rollout_initial_guess,
    simulate_monte_carlo,
)
from sooc_1dof_plot import (
    plot_monte_carlo_validation,
    plot_paper_style_comparison,
    plot_samples_behind_sd,
    plot_scenario_comparison,
)


def solve_scenario(
    *,
    label: str,
    N: int,
    dt: float,
    noise_std: float,
    variance_weight: float,
    excitation_guess: float,
    params,
):
    mean0 = np.zeros(2)
    covariance0 = np.zeros(3)
    activation0 = np.full(2, 0.05)

    opti, var = make_sooc_1dof_model(N=N, params=params)
    initial_guess = rollout_initial_guess(
        N=N,
        dt=dt,
        mean0=mean0,
        covariance0=covariance0,
        activation0=activation0,
        excitation_level=excitation_guess,
        noise_std=noise_std,
        params=params,
    )
    instantiate_sooc_1dof_model(
        var,
        opti,
        dt=dt,
        mean0=mean0,
        covariance0=covariance0,
        activation0=activation0,
        noise_std=noise_std,
        variance_weight=variance_weight,
        initial_guess=initial_guess,
    )

    opti.solver(
        "ipopt",
        {"expand": True, "print_time": True},
        {
            "max_iter": 3000,
            "tol": 1.0e-7,
            "acceptable_tol": 1.0e-5,
            "mu_strategy": "adaptive",
            "print_level": 5,
            "hessian_approximation": "limited-memory"
        },
    )

    try:
        solution = opti.solve()
    except RuntimeError:
        print(f"Optimization failed for scenario: {label}")
        opti.debug.show_infeasibilities(1.0e-6)
        raise

    numeric = numerize_var(var, solution)
    activation = np.asarray(numeric["variables"]["activation"], dtype=float)
    stiffness = np.asarray(numeric["functions"]["joint_stiffness"], dtype=float).reshape(-1)
    covariance = np.asarray(numeric["variables"]["covariance"], dtype=float)
    mean = np.asarray(numeric["variables"]["mean"], dtype=float)

    print("\nScenario:", label)
    print("  torque-noise SD [N m / sqrt(s)]:", noise_std)
    print("  variance weight:", variance_weight)
    print("  objective:", float(numeric["costs"]["objective"]))
    print("  effort cost:", float(numeric["costs"]["effort_cost"]))
    print("  running variance cost:", float(numeric["costs"]["variance_running_cost"]))
    print("  terminal variance cost:", float(numeric["costs"]["terminal_variance_cost"]))
    print("  maximum mean-angle magnitude [deg]:", np.rad2deg(np.max(np.abs(mean[0, :]))))
    print("  mean activation sum:", float(np.mean(activation.sum(axis=0))))
    print("  mean stiffness [N m/rad]:", float(np.mean(stiffness)))
    print("  final angle SD [deg]:", np.rad2deg(np.sqrt(max(covariance[0, -1], 0.0))))

    return {
        "label": label,
        "noise_std": noise_std,
        "variance_weight": variance_weight,
        "numeric": numeric,
    }


def main() -> None:
    params = default_sooc_parameters()
    total_time = 5.0
    N = 101
    dt = total_time / (N - 1)

    # The deterministic case has no state variance, so effort minimization does
    # not need co-contraction. With noise, increasing the variance weight should
    # make larger symmetric activation optimal.
    scenario_definitions = [
        {
            "label": "deterministic: no noise",
            "noise_std": 0.0,
            "variance_weight": 100.0,
            "excitation_guess": 0.05,
        },
        {
            "label": "stochastic: low variance penalty",
            "noise_std": 0.05,
            "variance_weight": 10.0,
            "excitation_guess": 0.50,
        },
        {
            "label": "stochastic: strong variance penalty",
            "noise_std": 0.05,
            "variance_weight": 100.0,
            "excitation_guess": 0.50,
        },
    ]

    scenarios = []
    for definition in scenario_definitions:
        scenarios.append(
            solve_scenario(
                N=N,
                dt=dt,
                params=params,
                **definition,
            )
        )

    plot_scenario_comparison(scenarios, params, dt)

    stochastic_sample_sets = []
    for scenario in scenarios[1:]:
        samples = simulate_monte_carlo(
            scenario["numeric"],
            dt=dt,
            noise_std=float(scenario["noise_std"]),
            n_samples=500,
            n_substeps=10,
            seed=1234,
        )
        stochastic_sample_sets.append({"label": scenario["label"], "samples": samples})

    plot_samples_behind_sd(stochastic_sample_sets, dt)
    plot_paper_style_comparison(
        scenarios[1:],
        stochastic_sample_sets,
        params,
        dt,
    )

    validation_scenario = scenarios[-1]
    validation_samples = stochastic_sample_sets[-1]["samples"]
    plot_monte_carlo_validation(
        validation_scenario["numeric"],
        validation_samples,
        dt,
        validation_scenario["label"],
    )


if __name__ == "__main__":
    main()
