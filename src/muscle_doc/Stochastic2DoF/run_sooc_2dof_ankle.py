#!/usr/bin/env python3
"""Run Step 3: two-DOF ankle stochastic optimal open-loop control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from sooc_2dof_ankle_func import (
    COV_DIM,
    NM,
    NQ,
    STATE_DIM,
    build_sooc_functions,
    default_sooc_parameters,
    instantiate_sooc_2dof_model,
    make_muscle_fun,
    make_pinocchio_functions,
    make_sooc_2dof_model,
    numeric_unpack_covariance,
    numerize_var,
    rollout_initial_guess,
    simulate_monte_carlo,
)
from sooc_2dof_ankle_plot import (
    plot_monte_carlo_validation,
    plot_scenario_comparison,
    plot_selectivity_summary,
)


def build_reduced_ankle_model(urdf_path: Path) -> pin.Model:
    """Load the 4-DOF test model and lock the two hip coordinates."""
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    reference_configuration = pin.neutral(full_model)
    locked_joint_names = ("hip_flexion", "hip_abduction")
    locked_joint_ids = []

    for name in locked_joint_names:
        joint_id = full_model.getJointId(name)
        if joint_id == 0:
            raise RuntimeError(f"Joint {name!r} was not found in {urdf_path}.")
        locked_joint_ids.append(joint_id)

    model = pin.buildReducedModel(
        full_model,
        locked_joint_ids,
        reference_configuration,
    )
    if model.nq != NQ or model.nv != NQ:
        raise RuntimeError(
            f"Expected a 2-DOF reduced ankle model, got nq={model.nq}, nv={model.nv}."
        )

    expected_names = ("ankle_flexion", "ankle_eversion")
    reduced_names = tuple(str(name) for name in model.names[1:])
    if reduced_names != expected_names:
        raise RuntimeError(
            f"Unexpected reduced joint order {reduced_names}; expected {expected_names}."
        )
    return model


def solve_scenario(
    *,
    label: str,
    N: int,
    dt: float,
    noise_std: Sequence[float],
    variance_weight: float,
    excitation_guess: Sequence[float],
    mean0: np.ndarray,
    covariance0: np.ndarray,
    activation0: np.ndarray,
    com_reference_xy: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    params: Mapping[str, Any],
    sooc_functions: Mapping[str, Any],
):
    opti, var = make_sooc_2dof_model(
        N=N,
        sooc_functions=sooc_functions,
        params=params,
    )

    initial_guess = rollout_initial_guess(
        rhs_fun=sooc_functions["rhs"],
        N=N,
        dt=dt,
        mean0=mean0,
        covariance0=covariance0,
        activation0=activation0,
        excitation_level=excitation_guess,
        noise_std=noise_std,
    )

    instantiate_sooc_2dof_model(
        var,
        opti,
        dt=dt,
        mean0=mean0,
        covariance0=covariance0,
        activation0=activation0,
        noise_std=noise_std,
        variance_weight=variance_weight,
        com_reference_xy=com_reference_xy,
        q_min=q_min,
        q_max=q_max,
        initial_guess=initial_guess,
    )

    opti.solver(
        "ipopt",
        {
            "expand": True,
            "print_time": True,
        },
        {
            "max_iter": 4000,
            "tol": 1.0e-7,
            "acceptable_tol": 1.0e-5,
            "mu_strategy": "adaptive",
            "print_level": 5,
        },
    )

    try:
        solution = opti.solve()
    except RuntimeError:
        print(f"Optimization failed for scenario: {label}")
        opti.debug.show_infeasibilities(1.0e-6)
        raise

    numeric = numerize_var(var, solution)
    mean = np.asarray(numeric["variables"]["mean"], dtype=float)
    covariance = np.asarray(numeric["variables"]["covariance"], dtype=float)
    activation_sum = np.asarray(
        numeric["functions"]["activation_sum"], dtype=float
    )
    stiffness = np.asarray(
        numeric["functions"]["joint_stiffness"], dtype=float
    )
    angle_std = np.asarray(
        numeric["functions"]["joint_angle_std"], dtype=float
    )
    com_std = np.asarray(numeric["functions"]["COM_std_xy"], dtype=float)

    start = N // 4
    minimum_covariance_eigenvalue = np.inf
    for node in range(N):
        matrix = numeric_unpack_covariance(covariance[:, node])
        minimum_covariance_eigenvalue = min(
            minimum_covariance_eigenvalue,
            float(np.linalg.eigvalsh(matrix).min()),
        )

    middle = N // 2
    mechanics = sooc_functions["mechanics"](
        mean[:, middle],
        np.asarray(numeric["variables"]["activation"], dtype=float)[:, middle],
    )
    mechanical_jacobian = np.asarray(mechanics[1], dtype=float)
    largest_local_real_eigenvalue = float(
        np.max(np.real(np.linalg.eigvals(mechanical_jacobian)))
    )

    print("\nScenario:", label)
    print("  torque-noise SD [N m / sqrt(s)]:", np.asarray(noise_std, dtype=float))
    print("  variance weight:", variance_weight)
    print("  objective:", float(numeric["costs"]["objective"]))
    print("  effort cost:", float(numeric["costs"]["effort_cost"]))
    print("  running variance cost:", float(numeric["costs"]["variance_running_cost"]))
    print("  terminal variance cost:", float(numeric["costs"]["terminal_variance_cost"]))
    print("  maximum mean-angle magnitude [deg]:", np.rad2deg(np.max(np.abs(mean[:NQ]))))
    print("  mean pair activation sums after transient:", activation_sum[:, start:].mean(axis=1))
    print("  mean joint stiffness after transient [N m/rad]:", stiffness[:, start:].mean(axis=1))
    print("  final joint-angle SD [deg]:", np.rad2deg(angle_std[:, -1]))
    print("  final COM SD [cm], [ML, AP]:", 100.0 * com_std[:, -1])
    print("  minimum covariance eigenvalue:", minimum_covariance_eigenvalue)
    print("  largest local mechanical eigenvalue at mid-horizon:", largest_local_real_eigenvalue)

    return {
        "label": label,
        "noise_std": np.asarray(noise_std, dtype=float),
        "variance_weight": float(variance_weight),
        "numeric": numeric,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    urdf_path = script_dir / "four_dof_ankle_hip.urdf"
    model = build_reduced_ankle_model(urdf_path)
    cmodel = cpin.Model(model)

    params = default_sooc_parameters()
    pin_functions = make_pinocchio_functions(cmodel)
    muscle_fun = make_muscle_fun(params)
    sooc_functions = build_sooc_functions(pin_functions, muscle_fun, params)

    total_time = 5.0
    N = 81
    dt = total_time / (N - 1)

    mean0 = np.zeros(STATE_DIM)
    covariance0 = np.zeros(COV_DIM)
    activation0 = np.full(NM, 0.05)
    q_min = np.asarray(model.lowerPositionLimit, dtype=float).reshape(NQ)
    q_max = np.asarray(model.upperPositionLimit, dtype=float).reshape(NQ)

    data = model.createData()
    reference_com = np.asarray(
        pin.centerOfMass(model, data, mean0[:NQ]), dtype=float
    ).reshape(3)
    com_reference_xy = reference_com[:2]
    gravity_torque = np.asarray(
        pin.computeGeneralizedGravity(model, data, mean0[:NQ]), dtype=float
    ).reshape(NQ)

    print("Reduced model joints:", tuple(str(name) for name in model.names[1:]))
    print("Reference COM:", reference_com)
    print("Gravity torque at upright reference [N m]:", gravity_torque)
    print("Covariance state size:", COV_DIM, "independent entries")

    # q0 is ankle flexion (sagittal/AP control); q1 is ankle eversion
    # (frontal/ML control). Directional-noise cases test whether the optimizer
    # selectively co-contracts the mechanically relevant antagonist pair.
    scenario_definitions = [
        # {
        #     "label": "deterministic: no noise",
        #     "noise_std": [0.0, 0.0],
        #     "variance_weight": 100.0,
        #     "excitation_guess": [0.05, 0.05, 0.05, 0.05],
        # },
        # {
        #     "label": "sagittal torque noise",
        #     "noise_std": [2.0, 0.0],
        #     "variance_weight": 100.0,
        #     "excitation_guess": [0.70, 0.70, 0.08, 0.08],
        # },
        # {
        #     "label": "frontal torque noise",
        #     "noise_std": [0.0, 2.0],
        #     "variance_weight": 100.0,
        #     "excitation_guess": [0.08, 0.08, 0.70, 0.70],
        # },
        {
            "label": "two-axis noise: low penalty",
            "noise_std": [2.0, 2.0],
            "variance_weight": 10.0,
            "excitation_guess": [0.50, 0.50, 0.50, 0.50],
        },
        {
            "label": "two-axis noise: strong penalty",
            "noise_std": [2.0, 2.0],
            "variance_weight": 100.0,
            "excitation_guess": [0.75, 0.75, 0.75, 0.75],
        },
    ]

    scenarios = []
    for definition in scenario_definitions:
        scenarios.append(
            solve_scenario(
                N=N,
                dt=dt,
                mean0=mean0,
                covariance0=covariance0,
                activation0=activation0,
                com_reference_xy=com_reference_xy,
                q_min=q_min,
                q_max=q_max,
                params=params,
                sooc_functions=sooc_functions,
                **definition,
            )
        )

    plot_scenario_comparison(scenarios, dt=dt)
    plot_selectivity_summary(scenarios)

    validation_scenario = scenarios[-1]
    validation_samples = simulate_monte_carlo(
        validation_scenario["numeric"],
        model=model,
        dt=dt,
        noise_std=validation_scenario["noise_std"],
        n_samples=300,
        n_substeps=5,
        seed=1234,
    )
    plot_monte_carlo_validation(
        validation_scenario,
        validation_samples,
        dt=dt,
    )


if __name__ == "__main__":
    main()
