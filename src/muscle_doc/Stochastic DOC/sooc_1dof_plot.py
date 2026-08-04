"""Plotting helpers for the 1-DOF SOOC validation problem."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_paper_style_comparison(
    scenarios: Sequence[Mapping[str, Any]],
    sample_sets: Sequence[Mapping[str, Any]],
    params: Mapping[str, float],
    dt: float,
    n_example_paths: int = 3,
) -> None:
    """Plot low- and strong-variance solutions in a paper-style A--D layout."""
    if len(scenarios) != 2 or len(sample_sets) != 2:
        raise ValueError("paper-style comparison requires exactly two stochastic conditions")

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 6.5),
        sharex="col",
        constrained_layout=True,
    )
    gravitational_stiffness = (
        float(params["mass"])
        * float(params["gravity"])
        * float(params["com_distance"])
    )
    torque_scale = float(params["moment_arm"]) * float(params["f_max"])

    for condition_index, (scenario, sample_set) in enumerate(zip(scenarios, sample_sets)):
        sample_label = str(sample_set["label"])
        if sample_label != str(scenario["label"]):
            raise ValueError("scenario and Monte Carlo sample labels must have the same order")

        samples = np.asarray(sample_set["samples"], dtype=float)
        if samples.ndim != 3 or samples.shape[1] != 2:
            raise ValueError("samples must have shape (n_samples, 2, n_time_nodes)")

        numeric = scenario["numeric"]
        activation = np.asarray(numeric["variables"]["activation"], dtype=float)
        stiffness = np.asarray(
            numeric["functions"]["joint_stiffness"], dtype=float
        ).reshape(-1)
        n_time_nodes = samples.shape[2]
        if activation.shape[1] != n_time_nodes or stiffness.size != n_time_nodes:
            raise ValueError("scenario and sample time dimensions do not match")

        t = np.arange(n_time_nodes) * dt
        sample_mean = samples.mean(axis=0)
        sample_std = samples.std(axis=0, ddof=1)
        state_column = 2 * condition_index
        mechanics_column = state_column + 1

        for row, state_index in enumerate((0, 1)):
            axis = axes[row, state_column]
            values = np.rad2deg(samples[:, state_index, :])
            mean = np.rad2deg(sample_mean[state_index, :])
            std = np.rad2deg(sample_std[state_index, :])
            for path in values[: min(n_example_paths, samples.shape[0])]:
                axis.plot(t, path, color="0.72", linewidth=0.8, alpha=0.75)
            axis.fill_between(t, mean - std, mean + std, color="0.75", alpha=0.55)
            axis.plot(t, mean, color="black", linewidth=2.0)
            axis.axhline(0.0, color="0.35", linewidth=0.7, linestyle=":")
            axis.set_ylabel("position [deg]" if row == 0 else "velocity [deg/s]")
            axis.grid(False)

        stiffness_axis = axes[0, mechanics_column]
        stiffness_axis.plot(t, stiffness, color="black", linewidth=2.0, label=r"$K(a_f+a_e)$")
        stiffness_axis.axhline(
            gravitational_stiffness,
            color="0.35",
            linewidth=1.8,
            linestyle=":",
            label=r"$mgl_c$",
        )
        stiffness_axis.set_ylabel("stiffness [N m/rad]")
        stiffness_axis.legend(frameon=False, fontsize="small")

        torque_axis = axes[1, mechanics_column]
        torque_axis.plot(
            t,
            torque_scale * activation[0, :],
            color="black",
            linewidth=2.0,
            label=r"flexor $u_1$",
        )
        torque_axis.plot(
            t,
            -torque_scale * activation[1, :],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label=r"extensor $u_2$",
        )
        torque_axis.axhline(
            gravitational_stiffness,
            color="0.35",
            linewidth=1.8,
            linestyle=":",
            label=r"$mgl_c$",
        )
        torque_axis.set_ylabel("muscle torque [N m]")
        torque_axis.legend(frameon=False, fontsize="small")

        short_label = "low penalty" if condition_index == 0 else "strong penalty"
        axes[0, state_column].set_title(short_label, fontsize="medium")
        axes[0, mechanics_column].set_title(short_label, fontsize="medium")

    for column in range(4):
        axes[1, column].set_xlabel("time [s]")
        for row in range(2):
            axes[row, column].spines[["top", "right"]].set_visible(False)

    # Use identical limits for quantities that are compared across conditions:
    # A versus C for kinematics, and B versus D for mechanics.
    comparable_axes = (
        (axes[0, 0], axes[0, 2]),  # position
        (axes[1, 0], axes[1, 2]),  # velocity
        (axes[0, 1], axes[0, 3]),  # stiffness
        (axes[1, 1], axes[1, 3]),  # antagonist muscle torque
    )
    for axis_left, axis_right in comparable_axes:
        lower = min(axis_left.get_ylim()[0], axis_right.get_ylim()[0])
        upper = max(axis_left.get_ylim()[1], axis_right.get_ylim()[1])
        axis_left.set_ylim(lower, upper)
        axis_right.set_ylim(lower, upper)

    for column, panel_letter in enumerate("ABCD"):
        axes[0, column].text(
            -0.20,
            1.08,
            panel_letter,
            transform=axes[0, column].transAxes,
            fontsize=24,
            va="top",
        )

    fig.suptitle("Stochastic postural control: low versus strong variability penalty")
    plt.show()


def plot_samples_behind_sd(
    sample_sets: Sequence[Mapping[str, Any]],
    dt: float,
) -> None:
    """Compare the Monte Carlo time histories used to compute sampled SDs.

    The sample standard deviation is evaluated independently at every time
    node over the first array dimension, using ``ddof=1``.
    """
    if not sample_sets:
        raise ValueError("at least one Monte Carlo sample set is required")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    quantities = (
        (0, "joint angle", "angle [deg]"),
        (1, "joint angular velocity", "angular velocity [deg/s]"),
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    expected_n_time_nodes = None

    for set_index, sample_set in enumerate(sample_sets):
        label = str(sample_set["label"])
        samples = np.asarray(sample_set["samples"], dtype=float)
        if samples.ndim != 3 or samples.shape[1] != 2:
            raise ValueError("samples must have shape (n_samples, 2, n_time_nodes)")
        if samples.shape[0] < 2:
            raise ValueError("at least two Monte Carlo samples are required")

        n_samples, _, n_time_nodes = samples.shape
        if expected_n_time_nodes is None:
            expected_n_time_nodes = n_time_nodes
        elif n_time_nodes != expected_n_time_nodes:
            raise ValueError("all sample sets must have the same number of time nodes")

        t = np.arange(n_time_nodes) * dt
        sampled_mean = samples.mean(axis=0)
        sampled_std = samples.std(axis=0, ddof=1)
        color = colors[set_index % len(colors)]

        for axis, (state_index, _, _) in zip(axes, quantities):
            values = np.rad2deg(samples[:, state_index, :])
            mean = np.rad2deg(sampled_mean[state_index, :])
            std = np.rad2deg(sampled_std[state_index, :])

            # Every faint line contributes to the pointwise sample SD. The
            # lines are rasterized so vector-graphics exports stay manageable.
            axis.plot(
                t,
                values.T,
                color=color,
                linewidth=0.40,
                alpha=0.025,
                rasterized=True,
            )
            axis.plot(t, mean, color=color, linewidth=2.0, label=f"{label}: mean")
            axis.fill_between(
                t,
                mean - std,
                mean + std,
                color=color,
                alpha=0.20,
                label=f"{label}: mean +/- 1 SD (n={n_samples})",
            )

    for axis, (_, title, ylabel) in zip(axes, quantities):
        axis.set_ylabel(ylabel)
        axis.set_title(f"Monte Carlo {title}: trajectories behind the pointwise SD")
        axis.grid(True)
        axis.legend(fontsize="small")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Low versus strong variability penalty")
    plt.show()


def plot_scenario_comparison(
    scenarios: Sequence[Mapping[str, Any]],
    params: Mapping[str, float],
    dt: float,
) -> None:
    """Compare optimized effort, impedance and predicted variability."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    gravitational_stiffness = (
        float(params["mass"])
        * float(params["gravity"])
        * float(params["com_distance"])
    )

    for scenario in scenarios:
        label = str(scenario["label"])
        numeric = scenario["numeric"]
        activation = np.asarray(numeric["variables"]["activation"], dtype=float)
        stiffness = np.asarray(numeric["functions"]["joint_stiffness"], dtype=float).reshape(-1)
        angle_std = np.asarray(numeric["functions"]["angle_std"], dtype=float).reshape(-1)
        mean = np.asarray(numeric["variables"]["mean"], dtype=float)
        t = np.arange(mean.shape[1]) * dt

        axes[0].plot(t, activation[0, :], label=f"flexor: {label}")
        axes[0].plot(t, activation[1, :], "--", label=f"extensor: {label}")
        axes[1].plot(t, activation.sum(axis=0), label=label)
        axes[2].plot(t, stiffness, label=label)
        axes[3].plot(t, np.rad2deg(angle_std), label=label)

    axes[0].set_ylabel("activation")
    axes[0].set_title("Antagonistic muscle activation")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True)
    axes[0].legend(ncol=2, fontsize="small")

    axes[1].set_ylabel(r"$a_f+a_e$")
    axes[1].set_title("Co-contraction level (activation sum)")
    axes[1].grid(True)
    axes[1].legend(fontsize="small")

    axes[2].axhline(
        gravitational_stiffness,
        linestyle=":",
        linewidth=1.5,
        label=r"destabilizing $mgl_c$",
    )
    axes[2].set_ylabel("stiffness [N m/rad]")
    axes[2].set_title("Activation-dependent joint stiffness")
    axes[2].grid(True)
    axes[2].legend(fontsize="small")

    axes[3].set_ylabel("angle SD [deg]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Predicted postural variability")
    axes[3].grid(True)
    axes[3].legend(fontsize="small")

    plt.show()


def plot_monte_carlo_validation(
    numeric: Mapping[str, Any],
    samples: np.ndarray,
    dt: float,
    label: str,
    n_paths: int = 20,
) -> None:
    """Compare propagated moments with sampled nonlinear trajectories."""
    mean = np.asarray(numeric["variables"]["mean"], dtype=float)
    covariance = np.asarray(numeric["variables"]["covariance"], dtype=float)
    activation = np.asarray(numeric["variables"]["activation"], dtype=float)
    excitation = np.asarray(numeric["variables"]["excitation"], dtype=float)
    stiffness = np.asarray(numeric["functions"]["joint_stiffness"], dtype=float).reshape(-1)

    N = mean.shape[1]
    t = np.arange(N) * dt
    tu = np.arange(N - 1) * dt
    predicted_angle_std = np.sqrt(np.maximum(covariance[0, :], 0.0))
    predicted_velocity_std = np.sqrt(np.maximum(covariance[2, :], 0.0))
    sampled_mean = samples.mean(axis=0)
    sampled_std = samples.std(axis=0, ddof=1)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)

    for path in samples[: min(n_paths, samples.shape[0])]:
        axes[0].plot(t, np.rad2deg(path[0, :]), linewidth=0.7, alpha=0.20)
    axes[0].plot(t, np.rad2deg(mean[0, :]), linewidth=2.0, label="predicted mean")
    axes[0].fill_between(
        t,
        np.rad2deg(mean[0, :] - 2.0 * predicted_angle_std),
        np.rad2deg(mean[0, :] + 2.0 * predicted_angle_std),
        alpha=0.20,
        label="predicted mean +/- 2 SD",
    )
    axes[0].plot(t, np.rad2deg(sampled_mean[0, :]), "--", label="Monte Carlo mean")
    axes[0].set_ylabel("angle [deg]")
    axes[0].set_title(f"SOOC moment validation: {label}")
    axes[0].grid(True)
    axes[0].legend(fontsize="small")

    axes[1].plot(t, np.rad2deg(predicted_angle_std), label="predicted angle SD")
    axes[1].plot(t, np.rad2deg(sampled_std[0, :]), "--", label="sampled angle SD")
    axes[1].plot(t, np.rad2deg(predicted_velocity_std), label="predicted velocity SD")
    axes[1].plot(t, np.rad2deg(sampled_std[1, :]), "--", label="sampled velocity SD")
    axes[1].set_ylabel("SD [deg or deg/s]")
    axes[1].grid(True)
    axes[1].legend(fontsize="small")

    axes[2].plot(t, activation[0, :], label="flexor activation")
    axes[2].plot(t, activation[1, :], "--", label="extensor activation")
    axes[2].plot(tu, excitation[0, :], ":", label="flexor excitation")
    axes[2].plot(tu, excitation[1, :], "-.", label="extensor excitation")
    axes[2].set_ylabel("activation / excitation")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(True)
    axes[2].legend(ncol=2, fontsize="small")

    axes[3].plot(t, stiffness, label="joint stiffness")
    axes[3].set_ylabel("N m/rad")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True)
    axes[3].legend(fontsize="small")

    plt.show()
