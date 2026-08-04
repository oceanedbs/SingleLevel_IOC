"""Plotting helpers for the Step-3 two-DOF ankle SOOC problem."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_scenario_comparison(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    dt: float,
) -> None:
    """Compare selective co-contraction and predicted variability."""
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for scenario_index, scenario in enumerate(scenarios):
        label = str(scenario["label"])
        numeric = scenario["numeric"]
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
        node_count = activation_sum.shape[1]
        t = np.arange(node_count) * dt
        color = colors[scenario_index % len(colors)]

        axes[0].plot(
            t,
            activation_sum[0],
            color=color,
            linewidth=1.8,
            label=f"flexion pair: {label}",
        )
        axes[0].plot(
            t,
            activation_sum[1],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"eversion pair: {label}",
        )

        axes[1].plot(
            t,
            stiffness[0],
            color=color,
            linewidth=1.8,
            label=f"flexion stiffness: {label}",
        )
        axes[1].plot(
            t,
            stiffness[1],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"eversion stiffness: {label}",
        )

        axes[2].plot(
            t,
            np.rad2deg(angle_std[0]),
            color=color,
            linewidth=1.8,
            label=f"flexion SD: {label}",
        )
        axes[2].plot(
            t,
            np.rad2deg(angle_std[1]),
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"eversion SD: {label}",
        )

        # Coordinate convention of the supplied URDF: x is ML and y is AP.
        axes[3].plot(
            t,
            100.0 * com_std[0],
            color=color,
            linewidth=1.8,
            label=f"COM ML SD: {label}",
        )
        axes[3].plot(
            t,
            100.0 * com_std[1],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"COM AP SD: {label}",
        )

    axes[0].set_ylabel("activation sum")
    axes[0].set_title("Antagonist-pair co-contraction")
    axes[0].grid(True)
    axes[0].legend(ncol=2, fontsize="x-small")

    axes[1].set_ylabel("N m/rad")
    axes[1].set_title("Activation-dependent ankle stiffness")
    axes[1].grid(True)
    axes[1].legend(ncol=2, fontsize="x-small")

    axes[2].set_ylabel("angle SD [deg]")
    axes[2].set_title("Predicted joint-angle variability")
    axes[2].grid(True)
    axes[2].legend(ncol=2, fontsize="x-small")

    axes[3].set_ylabel("COM SD [cm]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Predicted COM variability")
    axes[3].grid(True)
    axes[3].legend(ncol=2, fontsize="x-small")

    plt.show()


def plot_selectivity_summary(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    transient_fraction: float = 0.25,
) -> None:
    """Bar plots of steady co-contraction and COM variability by scenario."""
    if not 0.0 <= transient_fraction < 1.0:
        raise ValueError("transient_fraction must be in [0, 1).")

    labels = [str(scenario["label"]) for scenario in scenarios]
    pair_means = []
    com_std_means = []

    for scenario in scenarios:
        activation_sum = np.asarray(
            scenario["numeric"]["functions"]["activation_sum"], dtype=float
        )
        com_std = np.asarray(
            scenario["numeric"]["functions"]["COM_std_xy"], dtype=float
        )
        start = int(np.floor(transient_fraction * activation_sum.shape[1]))
        pair_means.append(activation_sum[:, start:].mean(axis=1))
        com_std_means.append(100.0 * com_std[:, start:].mean(axis=1))

    pair_means_array = np.asarray(pair_means)
    com_std_array = np.asarray(com_std_means)
    positions = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(positions - width / 2, pair_means_array[:, 0], width, label="flexion pair")
    axes[0].bar(positions + width / 2, pair_means_array[:, 1], width, label="eversion pair")
    axes[0].set_ylabel("mean activation sum")
    axes[0].set_title("Directional co-contraction selected by the optimizer")
    axes[0].set_xticks(positions, labels, rotation=15, ha="right")
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].legend()

    axes[1].bar(positions - width / 2, com_std_array[:, 0], width, label="COM ML")
    axes[1].bar(positions + width / 2, com_std_array[:, 1], width, label="COM AP")
    axes[1].set_ylabel("mean COM SD [cm]")
    axes[1].set_title("Resulting task-space variability")
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[1].legend()

    plt.show()


def plot_monte_carlo_validation(
    scenario: Mapping[str, Any],
    samples: Mapping[str, np.ndarray],
    *,
    dt: float,
    n_paths: int = 20,
) -> None:
    """Compare propagated joint/COM moments with nonlinear sampled trajectories."""
    numeric = scenario["numeric"]
    label = str(scenario["label"])
    mean = np.asarray(numeric["variables"]["mean"], dtype=float)
    predicted_angle_std = np.asarray(
        numeric["functions"]["joint_angle_std"], dtype=float
    )
    predicted_com_std = np.asarray(numeric["functions"]["COM_std_xy"], dtype=float)
    activation_sum = np.asarray(
        numeric["functions"]["activation_sum"], dtype=float
    )
    stiffness = np.asarray(
        numeric["functions"]["joint_stiffness"], dtype=float
    )

    state_samples = np.asarray(samples["state"], dtype=float)
    com_samples = np.asarray(samples["com_xy"], dtype=float)
    node_count = mean.shape[1]
    t = np.arange(node_count) * dt

    sampled_state_mean = state_samples.mean(axis=0)
    sampled_state_std = state_samples.std(axis=0, ddof=1)
    sampled_com_mean = com_samples.mean(axis=0)
    sampled_com_std = com_samples.std(axis=0, ddof=1)

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True, constrained_layout=True)

    for path in state_samples[: min(n_paths, state_samples.shape[0])]:
        axes[0].plot(t, np.rad2deg(path[0]), linewidth=0.6, alpha=0.15)
    axes[0].plot(t, np.rad2deg(mean[0]), linewidth=2.0, label="predicted mean flexion")
    axes[0].plot(
        t,
        np.rad2deg(sampled_state_mean[0]),
        "--",
        linewidth=1.8,
        label="sampled mean flexion",
    )
    axes[0].set_ylabel("angle [deg]")
    axes[0].set_title(f"Monte Carlo validation: {label}")
    axes[0].grid(True)
    axes[0].legend(fontsize="small")

    axes[1].plot(t, np.rad2deg(predicted_angle_std[0]), label="predicted flexion SD")
    axes[1].plot(t, np.rad2deg(sampled_state_std[0]), "--", label="sampled flexion SD")
    axes[1].plot(t, np.rad2deg(predicted_angle_std[1]), label="predicted eversion SD")
    axes[1].plot(t, np.rad2deg(sampled_state_std[1]), "--", label="sampled eversion SD")
    axes[1].set_ylabel("joint SD [deg]")
    axes[1].grid(True)
    axes[1].legend(ncol=2, fontsize="small")

    axes[2].plot(t, 100.0 * predicted_com_std[0], label="predicted COM ML SD")
    axes[2].plot(t, 100.0 * sampled_com_std[0], "--", label="sampled COM ML SD")
    axes[2].plot(t, 100.0 * predicted_com_std[1], label="predicted COM AP SD")
    axes[2].plot(t, 100.0 * sampled_com_std[1], "--", label="sampled COM AP SD")
    axes[2].set_ylabel("COM SD [cm]")
    axes[2].grid(True)
    axes[2].legend(ncol=2, fontsize="small")

    axes[3].plot(t, activation_sum[0], label="flexion-pair activation sum")
    axes[3].plot(t, activation_sum[1], "--", label="eversion-pair activation sum")
    axes[3].set_ylabel("activation sum")
    axes[3].grid(True)
    axes[3].legend(fontsize="small")

    axes[4].plot(t, stiffness[0], label="flexion stiffness")
    axes[4].plot(t, stiffness[1], "--", label="eversion stiffness")
    axes[4].set_ylabel("N m/rad")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(True)
    axes[4].legend(fontsize="small")

    plt.show()

    # Endpoint COM scatter provides a compact check of covariance orientation.
    final_com = com_samples[:, :, -1]
    predicted_com = np.asarray(numeric["functions"]["COM"], dtype=float)[:2, -1]
    plt.figure(figsize=(6.5, 6.0), constrained_layout=True)
    plt.scatter(
        100.0 * (final_com[:, 0] - predicted_com[0]),
        100.0 * (final_com[:, 1] - predicted_com[1]),
        s=8,
        alpha=0.25,
    )
    plt.axhline(0.0, linewidth=0.8, linestyle=":")
    plt.axvline(0.0, linewidth=0.8, linestyle=":")
    plt.xlabel("final COM ML error [cm]")
    plt.ylabel("final COM AP error [cm]")
    plt.title(f"Final sampled COM distribution: {label}")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
