"""CasADi/Pinocchio helpers for a deterministic muscle-driven DOC problem.

This is deliberately a reduced muscle model. Each joint is actuated by one
agonist/antagonist pair. Differential activation produces net joint torque;
the sum of the two activations increases local stiffness and damping.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import casadi as cs
import numpy as np
from pinocchio import casadi as cpin


def make_rnea_fun(cmodel: Any) -> cs.Function:
    """Return tau = RNEA(q, dq, ddq)."""
    q = cs.SX.sym("q", cmodel.nq)
    dq = cs.SX.sym("dq", cmodel.nv)
    ddq = cs.SX.sym("ddq", cmodel.nv)
    cdata = cpin.Data(cmodel)
    tau = cpin.rnea(cmodel, cdata, q, dq, ddq)
    return cs.Function("rnea_fun", [q, dq, ddq], [tau])


def make_com_fun(cmodel: Any) -> cs.Function:
    """Return whole-model COM as a function of q."""
    q = cs.SX.sym("q", cmodel.nq)
    cdata = cpin.Data(cmodel)
    com = cpin.centerOfMass(cmodel, cdata, q)
    return cs.Function("com_fun", [q], [com])


def default_muscle_parameters() -> Dict[str, Any]:
    """Parameters for eight abstract antagonist muscles.

    The model is intended as a first deterministic proof of concept, not as a
    subject-specific physiological parameter set.
    """
    return {
        "names": [
            "subtalar_positive",
            "subtalar_negative",
            "ankle_positive",
            "ankle_negative",
            "hip_rotation2_positive",
            "hip_rotation2_negative",
            "hip_rotation1_positive",
            "hip_rotation1_negative",
        ],
        "pair_names": [
            "subtalar_c pair",
            "ankle_c pair",
            "hip_c_rotation2 pair",
            "hip_c_rotation1 pair",
        ],
        "joint_names": [
            "ankle eversion (subtalar_c)",
            "ankle flexion (ankle_c)",
            "hip flexion (hip_c_rotation2)",
            "hip abduction (hip_c_rotation1)",
        ],
        # Muscle m acts around joint joint_index[m].
        "joint_index": np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int),
        # Positive direction is the positive coordinate direction in the URDF.
        "direction": np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float),
        "moment_arm": np.array([0.040, 0.050, 0.030, 0.030,
                                0.050, 0.050, 0.060, 0.060]),
        "f_max": np.array([2500.0, 4000.0, 1200.0, 1500.0,
                           3000.0, 4000.0, 3000.0, 3000.0]),
        # Activation-dependent short-range stiffness and damping parameters.
        "k_active": np.array([35000.0, 35000.0, 25000.0, 25000.0,
                              45000.0, 45000.0, 40000.0, 40000.0]),
        "b_active": np.array([1200.0, 1200.0, 800.0, 800.0,
                              1500.0, 1500.0, 1400.0, 1400.0]),
        "q_neutral": np.zeros(4),
        "k_passive": np.array([8.0, 5.0, 12.0, 10.0]),
        "b_passive": np.array([0.8, 0.5, 1.2, 1.0]),
        "tau_activation": np.full(8, 0.050),
        "pairs": [(0, 1), (2, 3), (4, 5), (6, 7)],
    }


def _as_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def validate_muscle_parameters(params: Mapping[str, Any], nq: int) -> None:
    names = list(params["names"])
    nm = len(names)
    if nm == 0:
        raise ValueError("At least one muscle is required.")

    for key in ("joint_index", "direction", "moment_arm", "f_max",
                "k_active", "b_active", "tau_activation"):
        if len(np.asarray(params[key]).reshape(-1)) != nm:
            raise ValueError(f"{key} must have one entry per muscle ({nm}).")

    joint_index = np.asarray(params["joint_index"], dtype=int).reshape(-1)
    if np.any(joint_index < 0) or np.any(joint_index >= nq):
        raise ValueError("joint_index contains an invalid joint number.")

    direction = _as_1d_float_array(params["direction"], "direction")
    if np.any(np.abs(np.abs(direction) - 1.0) > 1e-12):
        raise ValueError("Each muscle direction must be +1 or -1.")

    for key in ("moment_arm", "f_max", "k_active", "b_active",
                "tau_activation"):
        if np.any(_as_1d_float_array(params[key], key) <= 0.0):
            raise ValueError(f"{key} must be strictly positive.")

    for key in ("q_neutral", "k_passive", "b_passive"):
        if len(np.asarray(params[key]).reshape(-1)) != nq:
            raise ValueError(f"{key} must contain {nq} entries.")


def make_muscle_fun(nq: int, params: Mapping[str, Any]) -> cs.Function:
    """Create a reduced activation-dependent viscoelastic muscle model.

    For muscle m acting on joint j with signed moment arm s*r:

        dl_m     = -s*r*(q_j - q_neutral_j)
        d(dl_m)  = -s*r*dq_j
        F_m      = positive(Fmax*a + k*a*dl_m + b*a*d(dl_m))
        tau_m    = s*r*F_m

    Consequently, the differential activation of an antagonist pair generates
    net torque, while their summed activation increases joint stiffness and
    damping.
    """
    validate_muscle_parameters(params, nq)

    names = list(params["names"])
    nm = len(names)
    joint_index = np.asarray(params["joint_index"], dtype=int).reshape(-1)
    direction = np.asarray(params["direction"], dtype=float).reshape(-1)
    moment_arm = np.asarray(params["moment_arm"], dtype=float).reshape(-1)
    f_max = np.asarray(params["f_max"], dtype=float).reshape(-1)
    k_active = np.asarray(params["k_active"], dtype=float).reshape(-1)
    b_active = np.asarray(params["b_active"], dtype=float).reshape(-1)
    q_neutral = np.asarray(params["q_neutral"], dtype=float).reshape(-1)
    k_passive = np.asarray(params["k_passive"], dtype=float).reshape(-1)
    b_passive = np.asarray(params["b_passive"], dtype=float).reshape(-1)

    q = cs.SX.sym("q", nq)
    dq = cs.SX.sym("dq", nq)
    activation = cs.SX.sym("activation", nm)

    muscle_force = cs.SX.zeros(nm, 1)
    muscle_torque = cs.SX.zeros(nq, 1)
    joint_stiffness = cs.SX(k_passive)
    joint_damping = cs.SX(b_passive)

    # Smooth positive-part approximation. It only matters if a trial iterate
    # would otherwise produce compressive "muscle force".
    force_epsilon = 1e-4

    for m in range(nm):
        j = int(joint_index[m])
        s = float(direction[m])
        r = float(moment_arm[m])
        a_m = activation[m]

        delta_length = -s * r * (q[j] - float(q_neutral[j]))
        muscle_velocity = -s * r * dq[j]
        raw_force = (
            float(f_max[m]) * a_m
            + float(k_active[m]) * a_m * delta_length
            + float(b_active[m]) * a_m * muscle_velocity
        )
        positive_force = 0.5 * (
            raw_force + cs.sqrt(raw_force * raw_force + force_epsilon**2)
        )

        muscle_force[m] = positive_force
        muscle_torque[j] = muscle_torque[j] + s * r * positive_force
        joint_stiffness[j] = (
            joint_stiffness[j] + float(k_active[m]) * a_m * r * r
        )
        joint_damping[j] = (
            joint_damping[j] + float(b_active[m]) * a_m * r * r
        )

    passive_torque = -cs.SX(k_passive) * (q - cs.SX(q_neutral)) - cs.SX(b_passive) * dq
    total_torque = muscle_torque + passive_torque

    return cs.Function(
        "muscle_fun",
        [q, dq, activation],
        [total_torque, muscle_force, joint_stiffness, joint_damping],
        ["q", "dq", "activation"],
        ["tau", "force", "stiffness", "damping"],
    )


def _smooth_pair_overlap(a_1: cs.MX, a_2: cs.MX, epsilon: float = 1e-8) -> cs.MX:
    """Smooth approximation of min(a_1, a_2)."""
    return 0.5 * (a_1 + a_2 - cs.sqrt((a_1 - a_2) ** 2 + epsilon))


def make_muscle_driven_model(
    cmodel: Any,
    rnea_fun: cs.Function,
    com_fun: cs.Function,
    muscle_fun: cs.Function,
    muscle_params: Mapping[str, Any],
    N: int,
    weights: Mapping[str, float] | None = None,
) -> Tuple[cs.Opti, Dict[str, Any]]:
    """Build a deterministic direct-collocation problem.

    Decision variables are q, dq, ddq, muscle activation a, and neural
    excitation e. The core muscle-driven dynamics constraint is

        RNEA(q, dq, ddq) = tau_muscle(q, dq, a).
    """
    if N < 4:
        raise ValueError("N must be at least 4 to compute COM jerk.")
    if cmodel.nq != cmodel.nv:
        raise ValueError("This first implementation requires nq == nv.")

    nq = int(cmodel.nq)
    nm = len(muscle_params["names"])
    validate_muscle_parameters(muscle_params, nq)

    default_weights = {
        "activation": 1.0,
        "joint_torque": 1e-5,
        "joint_velocity": 2e-3,
        "joint_jerk": 1e-9,
        "com_velocity": 1e-2,
        "com_acceleration": 1e-4,
        "com_jerk": 1.0,
        "co_contraction": 1e-2,
    }
    if weights is not None:
        default_weights.update({key: float(value) for key, value in weights.items()})
    weights = default_weights

    opti = cs.Opti()
    var: Dict[str, Any] = {}

    parameters: Dict[str, Any] = {
        "dt": opti.parameter(),
        "q0": opti.parameter(nq),
        "dq0": opti.parameter(nq),
        "a0": opti.parameter(nm),
        "goal_COM_xy": opti.parameter(2),
        "q_min": opti.parameter(nq),
        "q_max": opti.parameter(nq),
    }
    var["parameters"] = parameters

    variables: Dict[str, Any] = {
        "q": opti.variable(nq, N),
        "dq": opti.variable(nq, N),
        "ddq": opti.variable(nq, N - 1),
        "activation": opti.variable(nm, N),
        "excitation": opti.variable(nm, N - 1),
    }
    var["variables"] = variables

    q = variables["q"]
    dq = variables["dq"]
    ddq = variables["ddq"]
    activation = variables["activation"]
    excitation = variables["excitation"]
    dt = parameters["dt"]

    tau_required_list = []
    tau_muscle_list = []
    force_list = []
    stiffness_list = []
    damping_list = []
    com_list = []

    for k in range(N):
        muscle_output = muscle_fun(q[:, k], dq[:, k], activation[:, k])
        tau_muscle_k = muscle_output[0]
        force_k = muscle_output[1]
        stiffness_k = muscle_output[2]
        damping_k = muscle_output[3]

        tau_muscle_list.append(tau_muscle_k)
        force_list.append(force_k)
        stiffness_list.append(stiffness_k)
        damping_list.append(damping_k)
        com_list.append(com_fun(q[:, k]))

        if k < N - 1:
            tau_required_list.append(rnea_fun(q[:, k], dq[:, k], ddq[:, k]))

    functions: Dict[str, Any] = {
        "tau_required": cs.horzcat(*tau_required_list),
        "muscle_tau": cs.horzcat(*tau_muscle_list),
        "muscle_force": cs.horzcat(*force_list),
        "joint_stiffness": cs.horzcat(*stiffness_list),
        "joint_damping": cs.horzcat(*damping_list),
        "COM": cs.horzcat(*com_list),
    }

    coactivation_rows = []
    for first, second in muscle_params["pairs"]:
        a_first = activation[first, :]
        a_second = activation[second, :]
        overlap = _smooth_pair_overlap(a_first, a_second)
        coactivation_rows.append(2.0 * overlap / (a_first + a_second + 1e-6))
    functions["coactivation"] = cs.vertcat(*coactivation_rows)
    functions["joint_jerk"] = (ddq[:, 1:] - ddq[:, :-1]) / dt
    functions["COM_velocity"] = (
        functions["COM"][:, 1:] - functions["COM"][:, :-1]
    ) / dt
    functions["COM_acceleration"] = (
        functions["COM_velocity"][:, 1:] - functions["COM_velocity"][:, :-1]
    ) / dt
    functions["COM_jerk"] = (
        functions["COM_acceleration"][:, 1:]
        - functions["COM_acceleration"][:, :-1]
    ) / dt
    var["functions"] = functions

    tau_activation = cs.repmat(
        cs.DM(np.asarray(muscle_params["tau_activation"], dtype=float).reshape(nm, 1)),
        1,
        N - 1,
    )

    constraints: Dict[str, Any] = {
        "initial_pos": q[:, 0] - parameters["q0"],
        "initial_vel": dq[:, 0] - parameters["dq0"],
        "initial_activation": activation[:, 0] - parameters["a0"],
        "dynamics_pos": q[:, 1:] - q[:, :-1] - dt * dq[:, :-1],
        "dynamics_vel": dq[:, 1:] - dq[:, :-1] - dt * ddq,
        "activation_dynamics": (
            activation[:, 1:]
            - activation[:, :-1]
            - dt * (excitation - activation[:, :-1]) / tau_activation
        ),
        "muscle_driven_dynamics": (
            functions["tau_required"] - functions["muscle_tau"][:, :-1]
        ),
        "com_xy_final": functions["COM"][:2, -1] - parameters["goal_COM_xy"],
        "final_velocity": dq[:, -1],
    }
    var["constraints"] = constraints

    for expression in constraints.values():
        opti.subject_to(expression == 0)

    q_min_matrix = cs.repmat(parameters["q_min"], 1, N)
    q_max_matrix = cs.repmat(parameters["q_max"], 1, N)
    # CasADi interprets inequalities between matrices as positive-semidefinite
    # constraints, which require square matrices. Vectorize these trajectories
    # to express the intended element-wise joint-position bounds.
    opti.subject_to(
        opti.bounded(cs.vec(q_min_matrix), cs.vec(q), cs.vec(q_max_matrix))
    )
    opti.subject_to(opti.bounded(0.0, activation, 1.0))
    opti.subject_to(opti.bounded(0.0, excitation, 1.0))
    opti.subject_to(opti.bounded(-20.0, dq, 20.0))
    opti.subject_to(opti.bounded(-100.0, ddq, 100.0))

    costs: Dict[str, Any] = {}
    costs["activation_cost"] = scaled_mean_square(
        activation,
        np.ones(nm),
    )

    costs["joint_torque_cost"] = scaled_mean_square(
        functions["tau_required"],
        np.array([50.0, 30.0, 75.0, 75.0]),
    )

    costs["joint_velocity_cost"] = scaled_mean_square(
        dq[:, :-1],
        np.array([1.0, 1.0, 1.0, 1.0]),
    )

    costs["joint_jerk_cost"] = scaled_mean_square(
        functions["joint_jerk"],
        np.array([10.0, 5.0, 15.0, 10.0]),
    )

    costs["com_velocity_cost"] = scaled_mean_square(
        functions["COM_velocity"],
        np.full(3, 0.1),
    )

    costs["com_acceleration_cost"] = scaled_mean_square(
        functions["COM_acceleration"],
        np.full(3, 0.3),
    )

    costs["com_jerk_cost"] = scaled_mean_square(
        functions["COM_jerk"],
        np.full(3, 1.0),
    )

    costs["co_contraction_cost"] = scaled_mean_square(
        functions["coactivation"],
        np.ones(len(muscle_params["pairs"])),
    )
    objective = (
        weights["activation"] * costs["activation_cost"]
        + weights["joint_torque"] * costs["joint_torque_cost"]
        + weights["joint_velocity"] * costs["joint_velocity_cost"]
        + weights["joint_jerk"] * costs["joint_jerk_cost"]
        + weights["com_velocity"] * costs["com_velocity_cost"]
        + weights["com_acceleration"] * costs["com_acceleration_cost"]
        + weights["com_jerk"] * costs["com_jerk_cost"]
        + weights["co_contraction"] * costs["co_contraction_cost"]
    )
    costs["objective"] = objective
    costs["weights"] = dict(weights)
    var["costs"] = costs

    opti.minimize(objective)
    return opti, var

def scaled_mean_square(expression, scale):
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    scale_matrix = cs.repmat(cs.DM(scale), 1, expression.shape[1])
    normalized = cs.times(expression, 1.0 / scale_matrix)
    return cs.sumsqr(normalized) / expression.numel()


def instantiate_muscle_driven_model(
    var: Mapping[str, Any],
    opti: cs.Opti,
    *,
    dt: float,
    q0: Sequence[float],
    dq0: Sequence[float],
    a0: Sequence[float],
    goal_com_xy: Sequence[float],
    q_min: Sequence[float],
    q_max: Sequence[float],
    q_guess: np.ndarray,
    dq_guess: np.ndarray,
    ddq_guess: np.ndarray,
    activation_guess: np.ndarray,
    excitation_guess: np.ndarray,
) -> None:
    """Set model parameters and initial guesses with shape validation."""
    parameters = var["parameters"]
    variables = var["variables"]

    opti.set_value(parameters["dt"], float(dt))
    opti.set_value(parameters["q0"], np.asarray(q0, dtype=float))
    opti.set_value(parameters["dq0"], np.asarray(dq0, dtype=float))
    opti.set_value(parameters["a0"], np.asarray(a0, dtype=float))
    goal_com_xy = np.asarray(goal_com_xy, dtype=float).reshape(-1)
    if goal_com_xy.shape != (2,):
        raise ValueError(
            f"goal_com_xy must contain exactly (x, y); got shape {goal_com_xy.shape}."
        )
    opti.set_value(parameters["goal_COM_xy"], goal_com_xy)
    opti.set_value(parameters["q_min"], np.asarray(q_min, dtype=float))
    opti.set_value(parameters["q_max"], np.asarray(q_max, dtype=float))

    guesses = {
        "q": np.asarray(q_guess, dtype=float),
        "dq": np.asarray(dq_guess, dtype=float),
        "ddq": np.asarray(ddq_guess, dtype=float),
        "activation": np.asarray(activation_guess, dtype=float),
        "excitation": np.asarray(excitation_guess, dtype=float),
    }
    for name, guess in guesses.items():
        expected_shape = variables[name].shape
        if guess.shape != expected_shape:
            raise ValueError(
                f"Initial guess {name} has shape {guess.shape}; expected {expected_shape}."
            )
        opti.set_initial(variables[name], guess)


def numerize_var(model_var: Mapping[str, Any], solution: Any) -> Dict[str, Any]:
    """Evaluate the numeric parts of the nested model dictionary."""
    numeric: Dict[str, Any] = {}
    for category_name, category in model_var.items():
        numeric[category_name] = {}
        for name, value in category.items():
            if category_name == "costs" and name == "weights":
                numeric[category_name][name] = dict(value)
            elif isinstance(value, (list, tuple)):
                numeric[category_name][name] = [solution.value(item) for item in value]
            elif isinstance(value, dict):
                numeric[category_name][name] = dict(value)
            else:
                numeric[category_name][name] = solution.value(value)
    return numeric
