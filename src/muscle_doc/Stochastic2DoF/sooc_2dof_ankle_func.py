"""Step 3: stochastic optimal open-loop control for a 2-DOF ankle model.

The stochastic mechanical state is

    x = [q_ankle_flexion, q_ankle_eversion,
         dq_ankle_flexion, dq_ankle_eversion].

The deterministic equivalent propagates the mean of x and the ten independent
entries of its 4 x 4 covariance matrix. Four deterministic muscle activations
(two antagonist pairs) act through the same reduced viscoelastic model used in
Step 1. Neural excitation remains the open-loop control.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

import casadi as cs
import numpy as np
from pinocchio import casadi as cpin


STATE_DIM = 4
NQ = 2
NM = 4
VECH_INDICES = tuple((row, col) for row in range(STATE_DIM) for col in range(row + 1))
COV_DIM = len(VECH_INDICES)
DIAGONAL_VECH_INDICES = tuple(VECH_INDICES.index((i, i)) for i in range(STATE_DIM))


def default_sooc_parameters() -> Dict[str, Any]:
    """Return numerical settings for the 2-DOF ankle validation problem."""
    return {
        # The muscle parameters are deliberately reduced proof-of-concept
        # values, chosen so activation-dependent stiffness can oppose the
        # instability of the upright two-link body in the supplied URDF.
        "muscle_names": [
            "ankle_flexor_positive",
            "ankle_flexor_negative",
            "ankle_evertor_positive",
            "ankle_evertor_negative",
        ],
        "pair_names": ["ankle flexion pair", "ankle eversion pair"],
        "joint_names": ["ankle flexion", "ankle eversion"],
        "joint_index": np.array([0, 0, 1, 1], dtype=int),
        "direction": np.array([1.0, -1.0, 1.0, -1.0]),
        "moment_arm": np.array([0.050, 0.050, 0.040, 0.040]),
        "f_max": np.array([5000.0, 5000.0, 4000.0, 4000.0]),
        "k_active": np.array([220000.0, 220000.0, 350000.0, 350000.0]),
        "b_active": np.array([100000.0, 100000.0, 160000.0, 160000.0]),
        "k_passive": np.array([20.0, 20.0]),
        "b_passive": np.array([5.0, 5.0]),
        "q_neutral": np.zeros(NQ),
        "tau_activation": np.full(NM, 0.050),
        "pairs": ((0, 1), (2, 3)),
        # Objective settings.
        "mean_com_weight": 1.0e5,
        "mean_velocity_weight": 1.0e3,
        "velocity_variance_scale": 0.01,
        "terminal_variance_scale": 0.20,
        "excitation_rate_weight": 1.0e-4,
        # Numerical covariance bounds. They are safeguards for NLP iterates,
        # not model predictions or hard scientific assumptions.
        "maximum_angle_variance": 1.0,
        "maximum_velocity_variance": 25.0,
    }


def validate_parameters(params: Mapping[str, Any]) -> None:
    required = (
        "muscle_names",
        "pair_names",
        "joint_names",
        "joint_index",
        "direction",
        "moment_arm",
        "f_max",
        "k_active",
        "b_active",
        "k_passive",
        "b_passive",
        "q_neutral",
        "tau_activation",
        "pairs",
        "mean_com_weight",
        "mean_velocity_weight",
        "velocity_variance_scale",
        "terminal_variance_scale",
        "excitation_rate_weight",
        "maximum_angle_variance",
        "maximum_velocity_variance",
    )
    for key in required:
        if key not in params:
            raise KeyError(f"Missing Step-3 parameter: {key}")

    if len(params["muscle_names"]) != NM:
        raise ValueError(f"Expected {NM} muscles.")
    if len(params["joint_names"]) != NQ:
        raise ValueError(f"Expected {NQ} joint names.")
    if len(params["pairs"]) != NQ:
        raise ValueError(f"Expected one antagonist pair per joint ({NQ}).")

    for key in (
        "joint_index",
        "direction",
        "moment_arm",
        "f_max",
        "k_active",
        "b_active",
        "tau_activation",
    ):
        values = np.asarray(params[key]).reshape(-1)
        if values.size != NM:
            raise ValueError(f"{key} must contain {NM} values.")
        if not np.all(np.isfinite(values.astype(float))):
            raise ValueError(f"{key} contains non-finite values.")

    for key in ("k_passive", "b_passive", "q_neutral"):
        values = np.asarray(params[key], dtype=float).reshape(-1)
        if values.size != NQ:
            raise ValueError(f"{key} must contain {NQ} values.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} contains non-finite values.")

    joint_index = np.asarray(params["joint_index"], dtype=int).reshape(-1)
    if np.any(joint_index < 0) or np.any(joint_index >= NQ):
        raise ValueError("joint_index contains an invalid joint number.")

    direction = np.asarray(params["direction"], dtype=float).reshape(-1)
    if np.any(np.abs(np.abs(direction) - 1.0) > 1.0e-12):
        raise ValueError("Each direction must be +1 or -1.")

    for key in ("moment_arm", "f_max", "k_active", "b_active", "tau_activation"):
        if np.any(np.asarray(params[key], dtype=float) <= 0.0):
            raise ValueError(f"{key} must be strictly positive.")


def _symmetrize_crba(mass_upper: cs.SX) -> cs.SX:
    return mass_upper + mass_upper.T - cs.diag(cs.diag(mass_upper))


def make_pinocchio_functions(cmodel: Any) -> Dict[str, cs.Function]:
    """Build CasADi functions for ABA, mass matrix and whole-model COM."""
    if int(cmodel.nq) != NQ or int(cmodel.nv) != NQ:
        raise ValueError(f"Expected a reduced 2-DOF model, got nq={cmodel.nq}, nv={cmodel.nv}.")

    q = cs.SX.sym("q", NQ)
    dq = cs.SX.sym("dq", NQ)
    tau = cs.SX.sym("tau", NQ)

    aba_data = cpin.Data(cmodel)
    ddq = cpin.aba(cmodel, aba_data, q, dq, tau)

    mass_data = cpin.Data(cmodel)
    mass_upper = cpin.crba(cmodel, mass_data, q)
    mass = _symmetrize_crba(mass_upper)

    com_data = cpin.Data(cmodel)
    com = cpin.centerOfMass(cmodel, com_data, q)

    return {
        "aba": cs.Function("ankle_aba", [q, dq, tau], [ddq]),
        "mass": cs.Function("ankle_mass", [q], [mass]),
        "com": cs.Function("ankle_com", [q], [com]),
    }


def make_muscle_fun(params: Mapping[str, Any]) -> cs.Function:
    """Return activation-dependent muscle torque, force and impedance.

    Muscle m, acting on joint j with signed moment arm s*r, follows

        dl_m    = -s*r*(q_j-q_neutral_j)
        dl_dot  = -s*r*dq_j
        F_m     = positive(Fmax*a + k*a*dl_m + b*a*dl_dot)
        tau_m   = s*r*F_m.

    The activation difference within a pair controls net torque; the activation
    sum increases joint stiffness and damping.
    """
    validate_parameters(params)

    joint_index = np.asarray(params["joint_index"], dtype=int).reshape(NM)
    direction = np.asarray(params["direction"], dtype=float).reshape(NM)
    moment_arm = np.asarray(params["moment_arm"], dtype=float).reshape(NM)
    f_max = np.asarray(params["f_max"], dtype=float).reshape(NM)
    k_active = np.asarray(params["k_active"], dtype=float).reshape(NM)
    b_active = np.asarray(params["b_active"], dtype=float).reshape(NM)
    k_passive = np.asarray(params["k_passive"], dtype=float).reshape(NQ)
    b_passive = np.asarray(params["b_passive"], dtype=float).reshape(NQ)
    q_neutral = np.asarray(params["q_neutral"], dtype=float).reshape(NQ)

    q = cs.SX.sym("q", NQ)
    dq = cs.SX.sym("dq", NQ)
    activation = cs.SX.sym("activation", NM)

    muscle_force = cs.SX.zeros(NM, 1)
    muscle_torque = cs.SX.zeros(NQ, 1)
    joint_stiffness = cs.SX(k_passive)
    joint_damping = cs.SX(b_passive)
    force_epsilon = 1.0e-4

    for muscle in range(NM):
        joint = int(joint_index[muscle])
        sign = float(direction[muscle])
        arm = float(moment_arm[muscle])
        a_m = activation[muscle]

        delta_length = -sign * arm * (q[joint] - float(q_neutral[joint]))
        muscle_velocity = -sign * arm * dq[joint]
        raw_force = (
            float(f_max[muscle]) * a_m
            + float(k_active[muscle]) * a_m * delta_length
            + float(b_active[muscle]) * a_m * muscle_velocity
        )
        positive_force = 0.5 * (
            raw_force + cs.sqrt(raw_force * raw_force + force_epsilon**2)
        )

        muscle_force[muscle] = positive_force
        muscle_torque[joint] += sign * arm * positive_force
        joint_stiffness[joint] += float(k_active[muscle]) * a_m * arm**2
        joint_damping[joint] += float(b_active[muscle]) * a_m * arm**2

    passive_torque = (
        -cs.SX(k_passive) * (q - cs.SX(q_neutral))
        - cs.SX(b_passive) * dq
    )
    total_torque = muscle_torque + passive_torque

    return cs.Function(
        "ankle_muscle",
        [q, dq, activation],
        [total_torque, muscle_force, joint_stiffness, joint_damping],
        ["q", "dq", "activation"],
        ["tau", "force", "stiffness", "damping"],
    )


def unpack_covariance(vector: cs.SX | cs.MX) -> cs.SX | cs.MX:
    """Convert the ten lower-triangular entries to a symmetric 4 x 4 matrix.

    Construct the result from symbolic expressions instead of assigning into
    an ``SX`` matrix. This preserves the input type when an Opti ``MX`` slice
    is supplied.
    """
    entry_index = {
        (row, col): index
        for index, (row, col) in enumerate(VECH_INDICES)
    }
    rows = []
    for row in range(STATE_DIM):
        entries = []
        for col in range(STATE_DIM):
            lower_index = entry_index[(max(row, col), min(row, col))]
            entries.append(vector[lower_index])
        rows.append(cs.horzcat(*entries))
    return cs.vertcat(*rows)


def pack_covariance(matrix: cs.SX | cs.MX) -> cs.SX | cs.MX:
    """Extract the lower-triangular entries of a symmetric 4 x 4 matrix."""
    return cs.vertcat(*[matrix[row, col] for row, col in VECH_INDICES])


def numeric_unpack_covariance(vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(COV_DIM)
    matrix = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    for index, (row, col) in enumerate(VECH_INDICES):
        matrix[row, col] = vector[index]
        matrix[col, row] = vector[index]
    return matrix


def numeric_pack_covariance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(STATE_DIM, STATE_DIM)
    return np.asarray([matrix[row, col] for row, col in VECH_INDICES], dtype=float)


def build_sooc_functions(
    pin_functions: Mapping[str, cs.Function],
    muscle_fun: cs.Function,
    params: Mapping[str, Any],
) -> Dict[str, cs.Function]:
    """Build the augmented mean/covariance/activation dynamics."""
    validate_parameters(params)

    aba_fun = pin_functions["aba"]
    mass_fun = pin_functions["mass"]
    com_fun = pin_functions["com"]

    mechanical_state = cs.SX.sym("mechanical_state", STATE_DIM)
    covariance_vector = cs.SX.sym("covariance_vector", COV_DIM)
    activation = cs.SX.sym("activation", NM)
    excitation = cs.SX.sym("excitation", NM)
    noise_std = cs.SX.sym("noise_std", NQ)

    q = mechanical_state[:NQ]
    dq = mechanical_state[NQ:]
    muscle_outputs = muscle_fun(q, dq, activation)
    muscle_tau = muscle_outputs[0]
    muscle_force = muscle_outputs[1]
    joint_stiffness = muscle_outputs[2]
    joint_damping = muscle_outputs[3]

    ddq = aba_fun(q, dq, muscle_tau)
    mechanical_rhs = cs.vertcat(dq, ddq)
    mechanical_jacobian = cs.jacobian(mechanical_rhs, mechanical_state)

    covariance = unpack_covariance(covariance_vector)
    mass = mass_fun(q)
    acceleration_noise_map = cs.solve(mass, cs.diag(noise_std))
    diffusion = cs.vertcat(cs.SX.zeros(NQ, NQ), acceleration_noise_map)
    covariance_dot = (
        mechanical_jacobian @ covariance
        + covariance @ mechanical_jacobian.T
        + diffusion @ diffusion.T
    )

    tau_activation = cs.SX(
        np.asarray(params["tau_activation"], dtype=float).reshape(NM)
    )
    activation_dot = (excitation - activation) / tau_activation

    augmented_state = cs.vertcat(mechanical_state, covariance_vector, activation)
    augmented_rhs = cs.vertcat(
        mechanical_rhs,
        pack_covariance(covariance_dot),
        activation_dot,
    )

    com = com_fun(q)
    com_jacobian = cs.jacobian(com, q)

    return {
        "rhs": cs.Function(
            "ankle_sooc_rhs",
            [augmented_state, excitation, noise_std],
            [augmented_rhs],
        ),
        "mechanics": cs.Function(
            "ankle_sooc_mechanics",
            [mechanical_state, activation],
            [
                mechanical_rhs,
                mechanical_jacobian,
                muscle_tau,
                muscle_force,
                joint_stiffness,
                joint_damping,
                com,
                com_jacobian,
            ],
            ["state", "activation"],
            [
                "state_dot",
                "A",
                "muscle_tau",
                "muscle_force",
                "joint_stiffness",
                "joint_damping",
                "com",
                "com_jacobian",
            ],
        ),
    }


def rk4_step(
    rhs_fun: cs.Function,
    state: cs.MX,
    excitation: cs.MX,
    dt: cs.MX,
    noise_std: cs.MX,
) -> cs.MX:
    k1 = rhs_fun(state, excitation, noise_std)
    k2 = rhs_fun(state + 0.5 * dt * k1, excitation, noise_std)
    k3 = rhs_fun(state + 0.5 * dt * k2, excitation, noise_std)
    k4 = rhs_fun(state + dt * k3, excitation, noise_std)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _trapezoidal_sum(values: cs.MX, dt: cs.MX) -> cs.MX:
    if values.shape[1] < 2:
        raise ValueError("At least two nodes are needed for trapezoidal integration.")
    return dt * cs.sum2(0.5 * (values[:, :-1] + values[:, 1:]))


def make_sooc_2dof_model(
    *,
    N: int,
    sooc_functions: Mapping[str, cs.Function],
    params: Mapping[str, Any],
) -> Tuple[cs.Opti, Dict[str, Any]]:
    """Build the equivalent deterministic 2-DOF SOOC problem."""
    if N < 4:
        raise ValueError("N must be at least 4.")
    validate_parameters(params)

    rhs_fun = sooc_functions["rhs"]
    mechanics_fun = sooc_functions["mechanics"]

    opti = cs.Opti()
    var: Dict[str, Any] = {}

    parameters: Dict[str, Any] = {
        "dt": opti.parameter(),
        "mean0": opti.parameter(STATE_DIM),
        "covariance0": opti.parameter(COV_DIM),
        "activation0": opti.parameter(NM),
        "noise_std": opti.parameter(NQ),
        "variance_weight": opti.parameter(),
        "com_reference_xy": opti.parameter(2),
        "q_min": opti.parameter(NQ),
        "q_max": opti.parameter(NQ),
    }
    var["parameters"] = parameters

    variables: Dict[str, Any] = {
        "mean": opti.variable(STATE_DIM, N),
        "covariance": opti.variable(COV_DIM, N),
        "activation": opti.variable(NM, N),
        "excitation": opti.variable(NM, N - 1),
    }
    var["variables"] = variables

    mean = variables["mean"]
    covariance_vector = variables["covariance"]
    activation = variables["activation"]
    excitation = variables["excitation"]
    dt = parameters["dt"]
    augmented_state = cs.vertcat(mean, covariance_vector, activation)

    constraints: Dict[str, Any] = {
        "initial_mean": mean[:, 0] - parameters["mean0"],
        "initial_covariance": covariance_vector[:, 0] - parameters["covariance0"],
        "initial_activation": activation[:, 0] - parameters["activation0"],
        "terminal_mean": mean[:, -1],
        # Postural task: prevent an artificial release at the final isolated
        # collocation node, which appeared in the Step-2 finite horizon result.
        "terminal_activation_steady": activation[:, -1] - activation[:, -2],
    }

    defects = []
    for node in range(N - 1):
        predicted = rk4_step(
            rhs_fun,
            augmented_state[:, node],
            excitation[:, node],
            dt,
            parameters["noise_std"],
        )
        defects.append(augmented_state[:, node + 1] - predicted)
    constraints["dynamics"] = cs.horzcat(*defects)
    var["constraints"] = constraints

    for expression in constraints.values():
        opti.subject_to(expression == 0)

    q_min_matrix = cs.repmat(parameters["q_min"], 1, N)
    q_max_matrix = cs.repmat(parameters["q_max"], 1, N)
    opti.subject_to(
        opti.bounded(cs.vec(q_min_matrix), cs.vec(mean[:NQ, :]), cs.vec(q_max_matrix))
    )
    opti.subject_to(opti.bounded(-5.0, mean[NQ:, :], 5.0))
    opti.subject_to(opti.bounded(0.0, activation, 1.0))
    opti.subject_to(opti.bounded(0.0, excitation, 1.0))

    # Covariance safeguards. The diagonal constraints and all 2 x 2 principal
    # minor inequalities stop the NLP from exploiting obviously unphysical
    # negative variances. Monte Carlo validation remains the definitive check.
    for node in range(N):
        covariance = unpack_covariance(covariance_vector[:, node])
        for state_index in range(STATE_DIM):
            opti.subject_to(covariance[state_index, state_index] >= 0.0)
        for first, second in combinations(range(STATE_DIM), 2):
            opti.subject_to(
                covariance[first, second] ** 2
                <= covariance[first, first] * covariance[second, second] + 1.0e-10
            )
        opti.subject_to(covariance[0, 0] <= float(params["maximum_angle_variance"]))
        opti.subject_to(covariance[1, 1] <= float(params["maximum_angle_variance"]))
        opti.subject_to(covariance[2, 2] <= float(params["maximum_velocity_variance"]))
        opti.subject_to(covariance[3, 3] <= float(params["maximum_velocity_variance"]))

    tau_columns = []
    force_columns = []
    stiffness_columns = []
    damping_columns = []
    com_columns = []
    com_covariance_columns = []
    angle_std_columns = []
    velocity_std_columns = []
    variance_density_columns = []

    for node in range(N):
        outputs = mechanics_fun(mean[:, node], activation[:, node])
        tau_columns.append(outputs[2])
        force_columns.append(outputs[3])
        stiffness_columns.append(outputs[4])
        damping_columns.append(outputs[5])
        com = outputs[6]
        com_jacobian = outputs[7]
        com_columns.append(com)

        covariance = unpack_covariance(covariance_vector[:, node])
        position_covariance = covariance[:NQ, :NQ]
        velocity_covariance = covariance[NQ:, NQ:]
        com_jacobian_xy = com_jacobian[:2, :]
        com_covariance_xy = (
            com_jacobian_xy @ position_covariance @ com_jacobian_xy.T
        )
        com_covariance_columns.append(
            cs.vertcat(
                com_covariance_xy[0, 0],
                com_covariance_xy[0, 1],
                com_covariance_xy[1, 1],
            )
        )

        angle_std_columns.append(
            cs.vertcat(
                cs.sqrt(cs.fmax(covariance[0, 0], 0.0) + 1.0e-16),
                cs.sqrt(cs.fmax(covariance[1, 1], 0.0) + 1.0e-16),
            )
        )
        velocity_std_columns.append(
            cs.vertcat(
                cs.sqrt(cs.fmax(covariance[2, 2], 0.0) + 1.0e-16),
                cs.sqrt(cs.fmax(covariance[3, 3], 0.0) + 1.0e-16),
            )
        )

        variance_density_columns.append(
            com_covariance_xy[0, 0]
            + com_covariance_xy[1, 1]
            + float(params["velocity_variance_scale"])
            * cs.trace(velocity_covariance)
        )

    functions: Dict[str, Any] = {
        "muscle_tau": cs.horzcat(*tau_columns),
        "muscle_force": cs.horzcat(*force_columns),
        "joint_stiffness": cs.horzcat(*stiffness_columns),
        "joint_damping": cs.horzcat(*damping_columns),
        "COM": cs.horzcat(*com_columns),
        "COM_covariance_xy": cs.horzcat(*com_covariance_columns),
        "joint_angle_std": cs.horzcat(*angle_std_columns),
        "joint_velocity_std": cs.horzcat(*velocity_std_columns),
        "activation_sum": cs.vertcat(
            activation[0, :] + activation[1, :],
            activation[2, :] + activation[3, :],
        ),
        "activation_difference": cs.vertcat(
            activation[0, :] - activation[1, :],
            activation[2, :] - activation[3, :],
        ),
    }
    functions["COM_std_xy"] = cs.vertcat(
        cs.sqrt(cs.fmax(functions["COM_covariance_xy"][0, :], 0.0) + 1.0e-16),
        cs.sqrt(cs.fmax(functions["COM_covariance_xy"][2, :], 0.0) + 1.0e-16),
    )
    var["functions"] = functions

    activation_energy_density = cs.sum1(activation**2) / NM
    effort_cost = _trapezoidal_sum(activation_energy_density, dt)

    com_error_xy = functions["COM"][:2, :] - cs.repmat(
        parameters["com_reference_xy"], 1, N
    )
    mean_com_density = cs.sum1(com_error_xy**2)
    mean_velocity_density = cs.sum1(mean[NQ:, :] ** 2)
    mean_cost = (
        float(params["mean_com_weight"]) * _trapezoidal_sum(mean_com_density, dt)
        + float(params["mean_velocity_weight"])
        * _trapezoidal_sum(mean_velocity_density, dt)
    )

    variance_density = cs.horzcat(*variance_density_columns)
    variance_running_cost = _trapezoidal_sum(variance_density, dt)
    terminal_variance_cost = float(params["terminal_variance_scale"]) * variance_density[0, -1]

    if N > 2:
        excitation_rate = (excitation[:, 1:] - excitation[:, :-1]) / dt
        excitation_rate_cost = (
            float(params["excitation_rate_weight"])
            * dt
            * cs.sumsqr(excitation_rate)
            / excitation_rate.numel()
        )
    else:
        excitation_rate_cost = cs.MX(0.0)

    objective = (
        effort_cost
        + mean_cost
        + parameters["variance_weight"]
        * (variance_running_cost + terminal_variance_cost)
        + excitation_rate_cost
    )

    costs: Dict[str, Any] = {
        "effort_cost": effort_cost,
        "mean_cost": mean_cost,
        "variance_running_cost": variance_running_cost,
        "terminal_variance_cost": terminal_variance_cost,
        "excitation_rate_cost": excitation_rate_cost,
        "objective": objective,
    }
    var["costs"] = costs
    var["model_parameters"] = dict(params)

    opti.minimize(objective)
    return opti, var


def rollout_initial_guess(
    *,
    rhs_fun: cs.Function,
    N: int,
    dt: float,
    mean0: Sequence[float],
    covariance0: Sequence[float],
    activation0: Sequence[float],
    excitation_level: Sequence[float],
    noise_std: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Generate a dynamically consistent RK4 initial guess."""
    if N < 2 or dt <= 0.0:
        raise ValueError("N must be >= 2 and dt must be positive.")

    state = np.zeros((STATE_DIM + COV_DIM + NM, N), dtype=float)
    state[:, 0] = np.concatenate(
        [
            np.asarray(mean0, dtype=float).reshape(STATE_DIM),
            np.asarray(covariance0, dtype=float).reshape(COV_DIM),
            np.asarray(activation0, dtype=float).reshape(NM),
        ]
    )
    excitation_level = np.asarray(excitation_level, dtype=float).reshape(NM)
    noise_std = np.asarray(noise_std, dtype=float).reshape(NQ)
    excitation = np.repeat(excitation_level[:, None], N - 1, axis=1)

    for node in range(N - 1):
        current = state[:, node]
        control = excitation[:, node]
        k1 = np.asarray(rhs_fun(current, control, noise_std), dtype=float).reshape(-1)
        k2 = np.asarray(
            rhs_fun(current + 0.5 * dt * k1, control, noise_std), dtype=float
        ).reshape(-1)
        k3 = np.asarray(
            rhs_fun(current + 0.5 * dt * k2, control, noise_std), dtype=float
        ).reshape(-1)
        k4 = np.asarray(rhs_fun(current + dt * k3, control, noise_std), dtype=float).reshape(-1)
        state[:, node + 1] = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Remove tiny integration roundoff from diagonal covariance entries.
        covariance_slice = state[STATE_DIM : STATE_DIM + COV_DIM, node + 1]
        for diagonal_index in DIAGONAL_VECH_INDICES:
            covariance_slice[diagonal_index] = max(covariance_slice[diagonal_index], 0.0)

    return {
        "mean": state[:STATE_DIM, :],
        "covariance": state[STATE_DIM : STATE_DIM + COV_DIM, :],
        "activation": state[STATE_DIM + COV_DIM :, :],
        "excitation": excitation,
    }


def instantiate_sooc_2dof_model(
    var: Mapping[str, Any],
    opti: cs.Opti,
    *,
    dt: float,
    mean0: Sequence[float],
    covariance0: Sequence[float],
    activation0: Sequence[float],
    noise_std: Sequence[float],
    variance_weight: float,
    com_reference_xy: Sequence[float],
    q_min: Sequence[float],
    q_max: Sequence[float],
    initial_guess: Mapping[str, np.ndarray],
) -> None:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    noise_std_array = np.asarray(noise_std, dtype=float).reshape(NQ)
    if np.any(noise_std_array < 0.0):
        raise ValueError("noise_std must be non-negative.")
    if variance_weight < 0.0:
        raise ValueError("variance_weight must be non-negative.")

    parameters = var["parameters"]
    opti.set_value(parameters["dt"], float(dt))
    opti.set_value(parameters["mean0"], np.asarray(mean0, dtype=float).reshape(STATE_DIM))
    opti.set_value(
        parameters["covariance0"], np.asarray(covariance0, dtype=float).reshape(COV_DIM)
    )
    opti.set_value(
        parameters["activation0"], np.asarray(activation0, dtype=float).reshape(NM)
    )
    opti.set_value(parameters["noise_std"], noise_std_array)
    opti.set_value(parameters["variance_weight"], float(variance_weight))
    opti.set_value(
        parameters["com_reference_xy"], np.asarray(com_reference_xy, dtype=float).reshape(2)
    )
    opti.set_value(parameters["q_min"], np.asarray(q_min, dtype=float).reshape(NQ))
    opti.set_value(parameters["q_max"], np.asarray(q_max, dtype=float).reshape(NQ))

    for name, variable in var["variables"].items():
        guess = np.asarray(initial_guess[name], dtype=float)
        expected_shape = tuple(int(value) for value in variable.shape)
        if guess.shape != expected_shape:
            raise ValueError(
                f"Initial guess {name} has shape {guess.shape}; expected {expected_shape}."
            )
        opti.set_initial(variable, guess)


def numerize_var(model_var: Mapping[str, Any], solution: Any) -> Dict[str, Any]:
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


def numeric_muscle_quantities(
    q: Sequence[float],
    dq: Sequence[float],
    activation: Sequence[float],
    params: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy counterpart of make_muscle_fun for Monte Carlo simulation."""
    q = np.asarray(q, dtype=float).reshape(NQ)
    dq = np.asarray(dq, dtype=float).reshape(NQ)
    activation = np.asarray(activation, dtype=float).reshape(NM)

    joint_index = np.asarray(params["joint_index"], dtype=int).reshape(NM)
    direction = np.asarray(params["direction"], dtype=float).reshape(NM)
    moment_arm = np.asarray(params["moment_arm"], dtype=float).reshape(NM)
    f_max = np.asarray(params["f_max"], dtype=float).reshape(NM)
    k_active = np.asarray(params["k_active"], dtype=float).reshape(NM)
    b_active = np.asarray(params["b_active"], dtype=float).reshape(NM)
    k_passive = np.asarray(params["k_passive"], dtype=float).reshape(NQ)
    b_passive = np.asarray(params["b_passive"], dtype=float).reshape(NQ)
    q_neutral = np.asarray(params["q_neutral"], dtype=float).reshape(NQ)

    forces = np.zeros(NM)
    torque = np.zeros(NQ)
    stiffness = k_passive.copy()
    damping = b_passive.copy()

    for muscle in range(NM):
        joint = int(joint_index[muscle])
        sign = direction[muscle]
        arm = moment_arm[muscle]
        delta_length = -sign * arm * (q[joint] - q_neutral[joint])
        muscle_velocity = -sign * arm * dq[joint]
        raw_force = (
            f_max[muscle] * activation[muscle]
            + k_active[muscle] * activation[muscle] * delta_length
            + b_active[muscle] * activation[muscle] * muscle_velocity
        )
        forces[muscle] = 0.5 * (raw_force + np.sqrt(raw_force**2 + 1.0e-8))
        torque[joint] += sign * arm * forces[muscle]
        stiffness[joint] += k_active[muscle] * activation[muscle] * arm**2
        damping[joint] += b_active[muscle] * activation[muscle] * arm**2

    torque += -k_passive * (q - q_neutral) - b_passive * dq
    return torque, forces, stiffness, damping


def simulate_monte_carlo(
    numeric: Mapping[str, Any],
    *,
    model: Any,
    dt: float,
    noise_std: Sequence[float],
    n_samples: int = 300,
    n_substeps: int = 5,
    seed: int = 1234,
) -> Dict[str, np.ndarray]:
    """Euler-Maruyama validation of the optimized open-loop activation."""
    import pinocchio as pin

    if n_samples < 2 or n_substeps < 1:
        raise ValueError("n_samples must be >= 2 and n_substeps must be >= 1.")

    mean = np.asarray(numeric["variables"]["mean"], dtype=float)
    activation = np.asarray(numeric["variables"]["activation"], dtype=float)
    params = numeric["model_parameters"]
    noise_std = np.asarray(noise_std, dtype=float).reshape(NQ)
    node_count = mean.shape[1]

    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, STATE_DIM, node_count), dtype=float)
    samples[:, :, 0] = mean[:, 0]
    com_samples = np.zeros((n_samples, 2, node_count), dtype=float)
    data_objects = [model.createData() for _ in range(n_samples)]

    for sample in range(n_samples):
        com_samples[sample, :, 0] = np.asarray(
            pin.centerOfMass(model, data_objects[sample], samples[sample, :NQ, 0]),
            dtype=float,
        ).reshape(3)[:2]

    h = dt / n_substeps
    sqrt_h = np.sqrt(h)

    for node in range(node_count - 1):
        state = samples[:, :, node].copy()
        for substep in range(n_substeps):
            alpha = (substep + 0.5) / n_substeps
            current_activation = (
                (1.0 - alpha) * activation[:, node]
                + alpha * activation[:, node + 1]
            )

            for sample in range(n_samples):
                q = state[sample, :NQ]
                dq = state[sample, NQ:]
                torque, _, _, _ = numeric_muscle_quantities(
                    q, dq, current_activation, params
                )
                data = data_objects[sample]
                ddq = np.asarray(pin.aba(model, data, q, dq, torque), dtype=float).reshape(NQ)
                mass = np.asarray(pin.crba(model, data, q), dtype=float)
                mass = np.triu(mass) + np.triu(mass, 1).T
                velocity_noise = np.linalg.solve(
                    mass,
                    noise_std * sqrt_h * rng.standard_normal(NQ),
                )
                dq_new = dq + h * ddq + velocity_noise
                q_new = q + h * dq_new
                state[sample, :NQ] = q_new
                state[sample, NQ:] = dq_new

        samples[:, :, node + 1] = state
        for sample in range(n_samples):
            com_samples[sample, :, node + 1] = np.asarray(
                pin.centerOfMass(
                    model,
                    data_objects[sample],
                    state[sample, :NQ],
                ),
                dtype=float,
            ).reshape(3)[:2]

    return {"state": samples, "com_xy": com_samples}
