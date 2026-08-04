"""Stochastic optimal open-loop control for a 1-DOF antagonistic muscle model.

This module implements the deterministic mean/covariance equivalent of a
stochastic open-loop control problem. The physical stochastic state is
x = [theta, omega]. Muscle activations and excitations remain deterministic.

The model is deliberately small: it is intended to validate the SOOC
formulation before adding covariance propagation to the 4-DOF model.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import casadi as cs
import numpy as np


def default_sooc_parameters() -> Dict[str, float]:
    """Return nominal parameters for an unstable upright forearm-like joint."""
    return {
        "inertia": 0.0588,          # kg m^2
        "mass": 1.44,               # kg
        "com_distance": 0.21,       # m
        "gravity": 9.81,            # m/s^2
        "moment_arm": 0.025,        # m
        "f_max": 800.0,             # N, same for the two abstract muscles
        "k_active": 6400.0,         # N/m per unit activation
        "b_active": 800.0,          # N s/m per unit activation
        "k_passive": 0.20,          # N m/rad
        "b_passive": 0.05,          # N m s/rad
        "tau_activation": 0.050,    # s
        "velocity_variance_scale": 0.01,
        "terminal_variance_scale": 1.0,
        "mean_state_weight": 1.0e4,
    }


def validate_parameters(params: Mapping[str, float]) -> None:
    required = (
        "inertia",
        "mass",
        "com_distance",
        "gravity",
        "moment_arm",
        "f_max",
        "k_active",
        "b_active",
        "k_passive",
        "b_passive",
        "tau_activation",
        "velocity_variance_scale",
        "terminal_variance_scale",
        "mean_state_weight",
    )
    for name in required:
        if name not in params:
            raise KeyError(f"Missing SOOC parameter: {name}")
        if not np.isfinite(float(params[name])):
            raise ValueError(f"Parameter {name} is not finite.")

    strictly_positive = (
        "inertia",
        "mass",
        "com_distance",
        "gravity",
        "moment_arm",
        "f_max",
        "k_active",
        "b_active",
        "tau_activation",
    )
    for name in strictly_positive:
        if float(params[name]) <= 0.0:
            raise ValueError(f"Parameter {name} must be strictly positive.")


def _mechanical_quantities(
    mean: cs.MX,
    activation: cs.MX,
    params: Mapping[str, float],
) -> Tuple[cs.MX, cs.MX, cs.MX, cs.MX, cs.MX]:
    """Return torque, stiffness, damping and the two mechanical Jacobian terms."""
    theta = mean[0]
    a_flexor = activation[0]
    a_extensor = activation[1]

    inertia = float(params["inertia"])
    mass = float(params["mass"])
    com_distance = float(params["com_distance"])
    gravity = float(params["gravity"])
    moment_arm = float(params["moment_arm"])
    f_max = float(params["f_max"])
    k_active = float(params["k_active"])
    b_active = float(params["b_active"])
    k_passive = float(params["k_passive"])
    b_passive = float(params["b_passive"])

    activation_sum = a_flexor + a_extensor
    activation_difference = a_flexor - a_extensor

    active_torque = moment_arm * f_max * activation_difference
    stiffness = k_passive + k_active * moment_arm**2 * activation_sum
    damping = b_passive + b_active * moment_arm**2 * activation_sum

    gravitational_stiffness = mass * gravity * com_distance
    a21 = (gravitational_stiffness * cs.cos(theta) - stiffness) / inertia
    a22 = -damping / inertia
    return active_torque, stiffness, damping, a21, a22


def augmented_rhs(
    state: cs.MX,
    excitation: cs.MX,
    noise_std: cs.MX,
    params: Mapping[str, float],
) -> cs.MX:
    """Dynamics of mean, covariance and deterministic muscle activation.

    State ordering:
        [mean_theta, mean_omega, P_theta_theta, P_theta_omega,
         P_omega_omega, a_flexor, a_extensor]

    The covariance dynamics use first-order statistical linearization:
        Pdot = A P + P A.T + G G.T.
    """
    mean = state[0:2]
    covariance = state[2:5]
    activation = state[5:7]

    theta = mean[0]
    omega = mean[1]
    p11 = covariance[0]
    p12 = covariance[1]
    p22 = covariance[2]

    inertia = float(params["inertia"])
    mass = float(params["mass"])
    com_distance = float(params["com_distance"])
    gravity = float(params["gravity"])
    tau_activation = float(params["tau_activation"])

    active_torque, stiffness, damping, a21, a22 = _mechanical_quantities(
        mean,
        activation,
        params,
    )

    mean_theta_dot = omega
    mean_omega_dot = (
        active_torque
        - stiffness * theta
        - damping * omega
        + mass * gravity * com_distance * cs.sin(theta)
    ) / inertia

    acceleration_noise_std = noise_std / inertia
    p11_dot = 2.0 * p12
    p12_dot = p22 + a21 * p11 + a22 * p12
    p22_dot = (
        2.0 * a21 * p12
        + 2.0 * a22 * p22
        + acceleration_noise_std**2
    )

    activation_dot = (excitation - activation) / tau_activation

    return cs.vertcat(
        mean_theta_dot,
        mean_omega_dot,
        p11_dot,
        p12_dot,
        p22_dot,
        activation_dot,
    )


def rk4_step(
    state: cs.MX,
    excitation: cs.MX,
    dt: cs.MX,
    noise_std: cs.MX,
    params: Mapping[str, float],
) -> cs.MX:
    """One Runge-Kutta fourth-order integration step."""
    k1 = augmented_rhs(state, excitation, noise_std, params)
    k2 = augmented_rhs(state + 0.5 * dt * k1, excitation, noise_std, params)
    k3 = augmented_rhs(state + 0.5 * dt * k2, excitation, noise_std, params)
    k4 = augmented_rhs(state + dt * k3, excitation, noise_std, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_sooc_1dof_model(
    N: int,
    params: Mapping[str, float] | None = None,
) -> Tuple[cs.Opti, Dict[str, Any]]:
    """Build the equivalent deterministic SOOC problem.

    The optimization minimizes effort plus a weighted state-variance cost.
    There is intentionally no direct co-contraction cost or constraint.
    """
    if N < 3:
        raise ValueError("N must be at least 3.")

    if params is None:
        params = default_sooc_parameters()
    params = dict(params)
    validate_parameters(params)

    opti = cs.Opti()
    var: Dict[str, Any] = {}

    parameters: Dict[str, Any] = {
        "dt": opti.parameter(),
        "mean0": opti.parameter(2),
        "covariance0": opti.parameter(3),
        "activation0": opti.parameter(2),
        "noise_std": opti.parameter(),
        "variance_weight": opti.parameter(),
    }
    var["parameters"] = parameters

    variables: Dict[str, Any] = {
        "mean": opti.variable(2, N),
        "covariance": opti.variable(3, N),
        "activation": opti.variable(2, N),
        "excitation": opti.variable(2, N - 1),
    }
    var["variables"] = variables

    mean = variables["mean"]
    covariance = variables["covariance"]
    activation = variables["activation"]
    excitation = variables["excitation"]
    dt = parameters["dt"]

    state = cs.vertcat(mean, covariance, activation)

    constraints: Dict[str, Any] = {
        "initial_mean": mean[:, 0] - parameters["mean0"],
        "initial_covariance": covariance[:, 0] - parameters["covariance0"],
        "initial_activation": activation[:, 0] - parameters["activation0"],
        "terminal_mean": mean[:, -1],
    }

    dynamic_defects = []
    for k in range(N - 1):
        predicted = rk4_step(
            state[:, k],
            excitation[:, k],
            dt,
            parameters["noise_std"],
            params,
        )
        dynamic_defects.append(state[:, k + 1] - predicted)
    constraints["dynamics"] = cs.horzcat(*dynamic_defects)
    var["constraints"] = constraints

    for expression in constraints.values():
        opti.subject_to(expression == 0)

    opti.subject_to(opti.bounded(0.0, activation, 1.0))
    opti.subject_to(opti.bounded(0.0, excitation, 1.0))
    opti.subject_to(opti.bounded(-0.35, mean[0, :], 0.35))
    opti.subject_to(opti.bounded(-4.0, mean[1, :], 4.0))

    # Continuous covariance propagation preserves positive semidefiniteness.
    # These light numerical safeguards prevent a direct-collocation iterate
    # from exploiting an unphysical negative variance.
    opti.subject_to(covariance[0, :] >= 0.0)
    opti.subject_to(covariance[2, :] >= 0.0)
    for k in range(N):
        determinant = (
            covariance[0, k] * covariance[2, k]
            - covariance[1, k] ** 2
        )
        opti.subject_to(determinant >= -1.0e-10)

    theta = mean[0, :]
    omega = mean[1, :]
    p11 = covariance[0, :]
    p22 = covariance[2, :]
    velocity_scale = float(params["velocity_variance_scale"])

    effort_cost = dt * cs.sumsqr(activation) / 2.0
    mean_state_cost = dt * cs.sum2(theta**2 + velocity_scale * omega**2)
    variance_running_cost = dt * cs.sum2(p11 + velocity_scale * p22)
    terminal_variance_cost = float(params["terminal_variance_scale"]) * (
        p11[0, N - 1] + velocity_scale * p22[0, N - 1]
    )

    objective = (
        effort_cost
        + float(params["mean_state_weight"]) * mean_state_cost
        + parameters["variance_weight"]
        * (variance_running_cost + terminal_variance_cost)
    )

    costs: Dict[str, Any] = {
        "effort_cost": effort_cost,
        "mean_state_cost": mean_state_cost,
        "variance_running_cost": variance_running_cost,
        "terminal_variance_cost": terminal_variance_cost,
        "objective": objective,
    }
    var["costs"] = costs

    torque_rows = []
    stiffness_rows = []
    damping_rows = []
    for k in range(N):
        active_torque, stiffness, damping, _, _ = _mechanical_quantities(
            mean[:, k],
            activation[:, k],
            params,
        )
        torque_rows.append(active_torque)
        stiffness_rows.append(stiffness)
        damping_rows.append(damping)

    functions: Dict[str, Any] = {
        "active_torque": cs.horzcat(*torque_rows),
        "joint_stiffness": cs.horzcat(*stiffness_rows),
        "joint_damping": cs.horzcat(*damping_rows),
        "activation_sum": activation[0, :] + activation[1, :],
        "activation_difference": activation[0, :] - activation[1, :],
        "angle_std": cs.sqrt(cs.fmax(covariance[0, :], 0.0) + 1.0e-16),
        "velocity_std": cs.sqrt(cs.fmax(covariance[2, :], 0.0) + 1.0e-16),
    }
    var["functions"] = functions
    var["model_parameters"] = params

    opti.minimize(objective)
    return opti, var


def _numeric_rhs(
    state: np.ndarray,
    excitation: np.ndarray,
    noise_std: float,
    params: Mapping[str, float],
) -> np.ndarray:
    """NumPy counterpart of augmented_rhs for initial-guess rollout."""
    theta, omega, p11, p12, p22, a_flexor, a_extensor = state

    inertia = float(params["inertia"])
    mass = float(params["mass"])
    com_distance = float(params["com_distance"])
    gravity = float(params["gravity"])
    moment_arm = float(params["moment_arm"])
    f_max = float(params["f_max"])
    k_active = float(params["k_active"])
    b_active = float(params["b_active"])
    k_passive = float(params["k_passive"])
    b_passive = float(params["b_passive"])
    tau_activation = float(params["tau_activation"])

    activation_sum = a_flexor + a_extensor
    activation_difference = a_flexor - a_extensor
    active_torque = moment_arm * f_max * activation_difference
    stiffness = k_passive + k_active * moment_arm**2 * activation_sum
    damping = b_passive + b_active * moment_arm**2 * activation_sum

    mean_theta_dot = omega
    mean_omega_dot = (
        active_torque
        - stiffness * theta
        - damping * omega
        + mass * gravity * com_distance * np.sin(theta)
    ) / inertia

    a21 = (
        mass * gravity * com_distance * np.cos(theta) - stiffness
    ) / inertia
    a22 = -damping / inertia
    acceleration_noise_std = noise_std / inertia

    p11_dot = 2.0 * p12
    p12_dot = p22 + a21 * p11 + a22 * p12
    p22_dot = (
        2.0 * a21 * p12
        + 2.0 * a22 * p22
        + acceleration_noise_std**2
    )

    activation_dot = (excitation - np.array([a_flexor, a_extensor])) / tau_activation
    return np.array(
        [
            mean_theta_dot,
            mean_omega_dot,
            p11_dot,
            p12_dot,
            p22_dot,
            activation_dot[0],
            activation_dot[1],
        ],
        dtype=float,
    )


def rollout_initial_guess(
    *,
    N: int,
    dt: float,
    mean0: Sequence[float],
    covariance0: Sequence[float],
    activation0: Sequence[float],
    excitation_level: float,
    noise_std: float,
    params: Mapping[str, float],
) -> Dict[str, np.ndarray]:
    """Generate a dynamically consistent symmetric initial guess."""
    state = np.zeros((7, N), dtype=float)
    state[:, 0] = np.concatenate(
        [
            np.asarray(mean0, dtype=float).reshape(2),
            np.asarray(covariance0, dtype=float).reshape(3),
            np.asarray(activation0, dtype=float).reshape(2),
        ]
    )
    excitation = np.full((2, N - 1), float(excitation_level), dtype=float)

    for k in range(N - 1):
        u = excitation[:, k]
        x = state[:, k]
        k1 = _numeric_rhs(x, u, noise_std, params)
        k2 = _numeric_rhs(x + 0.5 * dt * k1, u, noise_std, params)
        k3 = _numeric_rhs(x + 0.5 * dt * k2, u, noise_std, params)
        k4 = _numeric_rhs(x + dt * k3, u, noise_std, params)
        state[:, k + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Roundoff can produce tiny negative diagonal covariance entries.
        state[2, k + 1] = max(state[2, k + 1], 0.0)
        state[4, k + 1] = max(state[4, k + 1], 0.0)

    return {
        "mean": state[0:2, :],
        "covariance": state[2:5, :],
        "activation": state[5:7, :],
        "excitation": excitation,
    }


def instantiate_sooc_1dof_model(
    var: Mapping[str, Any],
    opti: cs.Opti,
    *,
    dt: float,
    mean0: Sequence[float],
    covariance0: Sequence[float],
    activation0: Sequence[float],
    noise_std: float,
    variance_weight: float,
    initial_guess: Mapping[str, np.ndarray],
) -> None:
    """Set parameter values and initial guesses."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if variance_weight < 0.0:
        raise ValueError("variance_weight must be non-negative.")

    parameters = var["parameters"]
    opti.set_value(parameters["dt"], float(dt))
    opti.set_value(parameters["mean0"], np.asarray(mean0, dtype=float).reshape(2))
    opti.set_value(
        parameters["covariance0"],
        np.asarray(covariance0, dtype=float).reshape(3),
    )
    opti.set_value(
        parameters["activation0"],
        np.asarray(activation0, dtype=float).reshape(2),
    )
    opti.set_value(parameters["noise_std"], float(noise_std))
    opti.set_value(parameters["variance_weight"], float(variance_weight))

    for name, variable in var["variables"].items():
        guess = np.asarray(initial_guess[name], dtype=float)
        expected = tuple(int(value) for value in variable.shape)
        if guess.shape != expected:
            raise ValueError(
                f"Initial guess {name} has shape {guess.shape}; expected {expected}."
            )
        opti.set_initial(variable, guess)


def numerize_var(model_var: Mapping[str, Any], solution: Any) -> Dict[str, Any]:
    """Evaluate the symbolic model dictionary at an Opti solution."""
    numeric: Dict[str, Any] = {}
    for category_name, category in model_var.items():
        if category_name == "model_parameters":
            numeric[category_name] = dict(category)
            continue
        numeric[category_name] = {}
        for name, value in category.items():
            if isinstance(value, dict):
                numeric[category_name][name] = dict(value)
            else:
                numeric[category_name][name] = solution.value(value)
    return numeric


def simulate_monte_carlo(
    numeric: Mapping[str, Any],
    *,
    dt: float,
    noise_std: float,
    n_samples: int = 500,
    n_substeps: int = 10,
    seed: int = 1234,
) -> np.ndarray:
    """Validate the moment propagation with Euler-Maruyama sampling.

    Returns an array of shape (n_samples, 2, N) containing angle and velocity.
    The optimized deterministic activation trajectory is linearly interpolated
    inside each collocation interval.
    """
    if n_samples <= 0 or n_substeps <= 0:
        raise ValueError("n_samples and n_substeps must be positive.")

    mean = np.asarray(numeric["variables"]["mean"], dtype=float)
    activation = np.asarray(numeric["variables"]["activation"], dtype=float)
    params = numeric["model_parameters"]
    N = mean.shape[1]

    inertia = float(params["inertia"])
    mass = float(params["mass"])
    com_distance = float(params["com_distance"])
    gravity = float(params["gravity"])
    moment_arm = float(params["moment_arm"])
    f_max = float(params["f_max"])
    k_active = float(params["k_active"])
    b_active = float(params["b_active"])
    k_passive = float(params["k_passive"])
    b_passive = float(params["b_passive"])

    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, 2, N), dtype=float)
    samples[:, :, 0] = mean[:, 0]
    h = dt / n_substeps
    noise_scale = noise_std / inertia * np.sqrt(h)

    for k in range(N - 1):
        state = samples[:, :, k].copy()
        for substep in range(n_substeps):
            alpha = (substep + 0.5) / n_substeps
            a = (1.0 - alpha) * activation[:, k] + alpha * activation[:, k + 1]
            activation_sum = a[0] + a[1]
            activation_difference = a[0] - a[1]
            active_torque = moment_arm * f_max * activation_difference
            stiffness = k_passive + k_active * moment_arm**2 * activation_sum
            damping = b_passive + b_active * moment_arm**2 * activation_sum

            theta = state[:, 0]
            omega = state[:, 1]
            acceleration = (
                active_torque
                - stiffness * theta
                - damping * omega
                + mass * gravity * com_distance * np.sin(theta)
            ) / inertia

            state[:, 0] = theta + h * omega
            state[:, 1] = omega + h * acceleration + noise_scale * rng.standard_normal(n_samples)

        samples[:, :, k + 1] = state

    return samples
