#!/usr/bin/env python3
"""Run deterministic 4-DOF muscle-driven direct optimal control."""

from __future__ import annotations

from pathlib import Path

import casadi as cs
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from scipy.optimize import least_squares

from muscle_doc_func import (
    default_muscle_parameters,
    instantiate_muscle_driven_model,
    make_com_fun,
    make_muscle_driven_model,
    make_muscle_fun,
    make_rnea_fun,
    numerize_var,
)
from muscle_doc_plot import plot_muscle_doc_results

def compute_lateral_q_from_com_xy(
    model,
    target_com_xy: np.ndarray,
) -> np.ndarray:
    """Compute q using only ankle eversion and hip abduction."""

    target_com_xy = np.asarray(
        target_com_xy,
        dtype=float,
    ).reshape(2)

    # Free: subtalar and hip abduction.
    free_indices = np.array([0, 3])

    # Ankle flexion and hip flexion remain zero.
    q_fixed = np.zeros(model.nq)

    q_min = np.asarray(
        model.lowerPositionLimit,
        dtype=float,
    )
    q_max = np.asarray(
        model.upperPositionLimit,
        dtype=float,
    )

    data = model.createData()

    def build_q(q_free):
        q = q_fixed.copy()
        q[free_indices] = q_free
        return q

    def com_error(q_free):
        q = build_q(q_free)
        com = np.asarray(
            pin.centerOfMass(model, data, q),
            dtype=float,
        ).reshape(3)

        return com[:2] - target_com_xy

    def com_jacobian(q_free):
        q = build_q(q_free)
        jacobian = np.asarray(
            pin.jacobianCenterOfMass(model, data, q),
            dtype=float,
        )

        return jacobian[:2, free_indices]

    result = least_squares(
        com_error,
        np.zeros(len(free_indices)),
        jac=com_jacobian,
        bounds=(
            q_min[free_indices],
            q_max[free_indices],
        ),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=1000,
    )

    q = build_q(result.x)
    error = com_error(result.x)

    if not result.success or np.max(np.abs(error)) > 1e-8:
        raise RuntimeError(
            "Could not reach the initial COM using only "
            f"ankle eversion and hip abduction; residual={error}."
        )

    return q

def smooth_posture_guess(q0: np.ndarray, q_goal: np.ndarray, N: int, dt: float):
    """Cubic interpolation with zero nominal endpoint velocity."""
    phase = np.linspace(0.0, 1.0, N)
    blend = 3.0 * phase**2 - 2.0 * phase**3
    q_guess = q0[:, None] + (q_goal - q0)[:, None] * blend[None, :]
    dq_guess = np.gradient(q_guess, dt, axis=1, edge_order=2)
    ddq_full = np.gradient(dq_guess, dt, axis=1, edge_order=2)
    ddq_guess = ddq_full[:, :-1]
    return q_guess, dq_guess, ddq_guess


def build_activation_guess(
    model,
    muscle_params,
    q_guess: np.ndarray,
    dq_guess: np.ndarray,
    ddq_guess: np.ndarray,
    dt: float,
):
    """Construct a torque-informed initial guess for antagonist activations."""
    nm = len(muscle_params["names"])
    N = q_guess.shape[1]
    activation = np.zeros((nm, N))

    direction = np.asarray(muscle_params["direction"], dtype=float)
    moment_arm = np.asarray(muscle_params["moment_arm"], dtype=float)
    f_max = np.asarray(muscle_params["f_max"], dtype=float)
    capacity = np.abs(direction * moment_arm * f_max)

    # Balanced low-level activity: each pair initially generates approximately
    # zero net torque despite unequal maximum strengths.
    baseline = np.zeros(nm)
    nominal = 0.05
    for first, second in muscle_params["pairs"]:
        baseline[first] = nominal
        baseline[second] = nominal * capacity[first] / capacity[second]
        pair_max = max(baseline[first], baseline[second])
        if pair_max > 0.15:
            baseline[[first, second]] *= 0.15 / pair_max

    activation[:] = baseline[:, None]
    data = model.createData()

    for k in range(N - 1):
        tau_required = np.asarray(
            pin.rnea(model, data, q_guess[:, k], dq_guess[:, k], ddq_guess[:, k]),
            dtype=float,
        ).reshape(model.nv)

        for joint, (first, second) in enumerate(muscle_params["pairs"]):
            current = (
                direction[first] * capacity[first] * activation[first, k]
                + direction[second] * capacity[second] * activation[second, k]
            )
            residual = tau_required[joint] - current
            if residual >= 0.0:
                positive = first if direction[first] > 0.0 else second
                activation[positive, k] += residual / capacity[positive]
            else:
                negative = first if direction[first] < 0.0 else second
                activation[negative, k] += (-residual) / capacity[negative]

        activation[:, k] = np.clip(activation[:, k], 0.001, 0.95)

    activation[:, -1] = activation[:, -2]
    tau_activation = np.asarray(muscle_params["tau_activation"], dtype=float)[:, None]
    activation_rate = np.diff(activation, axis=1) / dt
    excitation = activation[:, :-1] + tau_activation * activation_rate
    excitation = np.clip(excitation, 0.001, 0.99)
    return activation, excitation


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    urdf_path = script_dir / "subject_model.urdf"

    model = pin.buildModelFromUrdf(str(urdf_path))
    if model.nq != 4 or model.nv != 4:
        raise RuntimeError(f"Expected a 4-DOF fixed-base model, got nq={model.nq}, nv={model.nv}.")

    cmodel = cpin.Model(model)
    rnea_fun = make_rnea_fun(cmodel)
    com_fun = make_com_fun(cmodel)

    muscle_params = default_muscle_parameters()
    muscle_fun = make_muscle_fun(model.nq, muscle_params)

    N = 81
    total_time = 1.20
    dt = total_time / (N - 1)

    com_start_xy = np.array([ 0.05126017, -0.07352796])
    q0 = compute_lateral_q_from_com_xy(model, com_start_xy)
    dq0 = np.zeros(model.nv)
    target_com_xy = np.array([-0.04873983, -0.07352796])
    q_goal = compute_lateral_q_from_com_xy(model, target_com_xy)

    # reset goal COM to match the computed q_goal (in case of small numerical error)
    data = model.createData()
    initial_com = np.asarray(
        pin.centerOfMass(model, data, q0, dq0), dtype=float
    ).reshape(3)
    pin.forwardKinematics(model, data, q_goal)
    pin.updateFramePlacements(model, data)
    reference_goal_com = np.asarray(
        pin.centerOfMass(model, data, q_goal, dq0), dtype=float
    ).reshape(3)

    q_min = np.asarray(model.lowerPositionLimit, dtype=float).reshape(model.nq)
    q_max = np.asarray(model.upperPositionLimit, dtype=float).reshape(model.nq)

    q_guess, dq_guess, ddq_guess = smooth_posture_guess(q0, q_goal, N, dt)

    activation_guess, excitation_guess = build_activation_guess(
        model, muscle_params, q_guess, dq_guess, ddq_guess, dt
    )
    a0 = activation_guess[:, 0].copy()

    weights = {
        "activation": 1.0,
        "joint_torque": 2.0,
        "joint_velocity": 0.50,
        "joint_jerk": 1.0,
        "com_velocity": 0.50,
        "com_acceleration": 1.0,
        "com_jerk": 1.0,
        "co_contraction": 1.0,
    }

    opti, var = make_muscle_driven_model(
        cmodel=cmodel,
        rnea_fun=rnea_fun,
        com_fun=com_fun,
        muscle_fun=muscle_fun,
        muscle_params=muscle_params,
        N=N,
        weights=weights,
    )

    instantiate_muscle_driven_model(
        var,
        opti,
        dt=dt,
        q0=q0,
        dq0=dq0,
        a0=a0,
        goal_com_xy=target_com_xy,
        q_min=q_min,
        q_max=q_max,
        q_guess=q_guess,
        dq_guess=dq_guess,
        ddq_guess=ddq_guess,
        activation_guess=activation_guess,
        excitation_guess=excitation_guess,
    )

    opti.solver(
        "ipopt",
        {
            "expand": True,
            "print_time": True,
        },
        {
            "max_iter": 3000,
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
            "mu_strategy": "adaptive",
            "print_level": 5,
        },
    )

    try:
        solution = opti.solve()
    except RuntimeError:
        # Preserve IPOPT diagnostics and make the current iterate inspectable.
        print("Optimization failed. Current constraint violation:")
        opti.debug.show_infeasibilities(1e-5)
        raise

    numeric = numerize_var(var, solution)

    q_solution = np.asarray(numeric["variables"]["q"], dtype=float)
    tau_required = np.asarray(numeric["functions"]["tau_required"], dtype=float)
    tau_muscle = np.asarray(numeric["functions"]["muscle_tau"], dtype=float)[:, :-1]
    com_solution = np.asarray(numeric["functions"]["COM"], dtype=float)

    print("\nSolved deterministic muscle-driven DOC")
    print("Requested initial COM (x, y):", com_start_xy)
    print("Computed q0:", q0)
    print("Initial COM:", com_solution[:, 0])
    print("Final q:", q_solution[:, -1])
    print("Requested target COM (x, y):", target_com_xy)
    print("Computed q_goal:", q_goal)
    print("Final COM:", com_solution[:, -1])
    print("Target COM (x, y):", target_com_xy)
    print("Unconstrained initial COM z:", initial_com[2])
    print("Unconstrained reference COM z:", reference_goal_com[2])
    print("Maximum |RNEA - muscle torque|:", np.max(np.abs(tau_required - tau_muscle)))
    print("Objective:", float(numeric["costs"]["objective"]))
    for cost_name, cost_value in numeric["costs"].items():
        if cost_name.endswith("_cost"):
            print(f"{cost_name}:", float(cost_value))

    plot_muscle_doc_results(numeric, muscle_params, dt, target_com_xy)


if __name__ == "__main__":
    main()
