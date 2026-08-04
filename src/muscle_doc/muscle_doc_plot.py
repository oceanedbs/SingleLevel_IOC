"""Plotting helpers for the deterministic muscle-driven DOC example."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_muscle_doc_results(
    numeric: Mapping[str, Any],
    muscle_params: Mapping[str, Any],
    dt: float,
    goal_com_xy: Sequence[float],
) -> None:
    q = np.asarray(numeric["variables"]["q"], dtype=float)
    dq = np.asarray(numeric["variables"]["dq"], dtype=float)
    ddq = np.asarray(numeric["variables"]["ddq"], dtype=float)
    activation = np.asarray(numeric["variables"]["activation"], dtype=float)
    excitation = np.asarray(numeric["variables"]["excitation"], dtype=float)

    tau_required = np.asarray(numeric["functions"]["tau_required"], dtype=float)
    muscle_tau = np.asarray(numeric["functions"]["muscle_tau"], dtype=float)
    muscle_force = np.asarray(numeric["functions"]["muscle_force"], dtype=float)
    stiffness = np.asarray(numeric["functions"]["joint_stiffness"], dtype=float)
    com = np.asarray(numeric["functions"]["COM"], dtype=float)
    coactivation = np.asarray(numeric["functions"]["coactivation"], dtype=float)

    joint_names = list(muscle_params["joint_names"])
    muscle_names = list(muscle_params["names"])
    pair_names = list(muscle_params["pair_names"])

    t = np.arange(q.shape[1]) * dt
    t_interval = np.arange(ddq.shape[1]) * dt

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=False, constrained_layout=True)

    for j, name in enumerate(joint_names):
        axes[0].plot(t, q[j, :], label=name)
    axes[0].set_ylabel("q [rad]")
    axes[0].set_title("Joint trajectories")
    axes[0].grid(True)
    axes[0].legend(ncol=2, fontsize="small")

    for j, name in enumerate(joint_names):
        axes[1].plot(t_interval, tau_required[j, :], label=f"required: {name}")
        axes[1].plot(
            t_interval,
            muscle_tau[j, :-1],
            "--",
            label=f"muscle: {name}",
        )
    axes[1].set_ylabel("torque [N m]")
    axes[1].set_title("RNEA torque and muscle torque (curves should overlap)")
    axes[1].grid(True)
    axes[1].legend(ncol=2, fontsize="x-small")

    for m, name in enumerate(muscle_names):
        axes[2].plot(t, activation[m, :], label=f"a: {name}")
        axes[2].plot(t_interval, excitation[m, :], ":", label=f"e: {name}")
    axes[2].set_ylabel("activation / excitation")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Muscle states and controls")
    axes[2].grid(True)
    axes[2].legend(ncol=2, fontsize="xx-small")

    for j, name in enumerate(joint_names):
        axes[3].plot(t, stiffness[j, :], label=name)
    axes[3].set_ylabel("stiffness [N m/rad]")
    axes[3].set_title("Activation-dependent joint stiffness")
    axes[3].grid(True)
    axes[3].legend(ncol=2, fontsize="small")

    for axis, label in enumerate(("COM x (ML)", "COM y (AP)", "COM z")):
        axes[4].plot(t, com[axis, :], label=label)
        if axis < 2:
            axes[4].axhline(
                float(goal_com_xy[axis]), linestyle=":", linewidth=1.0
            )
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("COM [m]")
    axes[4].set_title("Whole-body COM")
    axes[4].grid(True)
    axes[4].legend(fontsize="small")

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for m, name in enumerate(muscle_names):
        axes2[0].plot(t, muscle_force[m, :], label=name)
    axes2[0].set_ylabel("force [N]")
    axes2[0].set_title("Muscle forces")
    axes2[0].grid(True)
    axes2[0].legend(ncol=2, fontsize="x-small")

    for pair, name in enumerate(pair_names):
        axes2[1].plot(t, coactivation[pair, :], label=name)
    axes2[1].set_xlabel("time [s]")
    axes2[1].set_ylabel("coactivation index")
    axes2[1].set_ylim(-0.02, 1.02)
    axes2[1].set_title("Normalized antagonist overlap")
    axes2[1].grid(True)
    axes2[1].legend(ncol=2, fontsize="small")

    plt.show()
