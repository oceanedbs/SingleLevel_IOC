import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_snapshots_from_vars(fig, var, nsnapshots=10):
    """
    Plots 3D snapshots of the robot in a single figure.

    Args:
        var (dict): Dictionary containing variables and functions (same as in the model)
        nsnapshots (int): Number of snapshots to display along the trajectory
    """
    q = var["variables"]["q"]
    n, N = q.shape  # number of joints, timesteps
    ax = fig.add_subplot(111, projection='3d')

    # -----------------------------
    # End-effector trajectory
    # -----------------------------
    P_end = var["functions"]["P"][-1]  # 3×N array
    ax.plot(P_end[0, :], P_end[1, :], P_end[2, :],
            color='y', linewidth=2, label="EE traj")

    # -----------------------------
    # Segment COM trajectories
    # -----------------------------
    for ii in range(n):
        Pcom = var["functions"]["Pcom"][ii]
        ax.plot(Pcom[0, :], Pcom[1, :], Pcom[2, :],
                color='c', linewidth=2, alpha=0.2)
        
    print('Pcom :', var['functions']['Pcom'])
    print('P : ', var['functions']['P'])

    # -----------------------------
    # Total COM trajectory (if exists)
    # -----------------------------
    # if "Pcomtotal" in var["functions"]:
    #     Pcomtotal = var["functions"]["Pcomtotal"]
    #     ax.plot(Pcomtotal[0, :], Pcomtotal[1, :], Pcomtotal[2, :],
    #             color=[0.2, 0.7, 0.05], linewidth=2, label="Total COM")

    # -----------------------------
    # Plot snapshots of the robot
    # -----------------------------
    snapshot_indices = np.floor(np.linspace(0, N-1, nsnapshots)).astype(int)

    for ii in snapshot_indices:
        P, _, _, _ = snapshot_from_vars(var, ii, ax=ax)
        alpha = 0.2
        dP = P[-1, :] - P[-2, :]
        ax.text(P[-1, 0] + alpha * dP[0],
                P[-1, 1] + alpha * dP[1],
                P[-1, 2] + alpha * dP[2],
                f"{ii}", fontsize=12, color='k')

    # Styling
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # equal scale
    plt.tight_layout()


def snapshot_from_vars(var, ii, ax=None, marker_size=6, line_width=2):
    """
    Draws a 3D snapshot of the robot at timestep ii.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    q = var["variables"]["q"]
    n = q.shape[0]

    # -----------------------------
    # Link positions
    # -----------------------------
    P = np.vstack([np.array(Pi) for Pi in var["functions"]["P"]])  # shape (3*(n+1), N)
    Px = P[0::3, ii]
    Py = P[1::3, ii]
    Pz = P[2::3, ii]

    # -----------------------------
    # COM positions
    # -----------------------------
    Pcom = np.hstack(var["functions"]["Pcom"])
    Pcomx = Pcom[0::3, ii]
    Pcomy = Pcom[1::3, ii]
    Pcomz = Pcom[2::3, ii]

    # Optional total COM
    # has_Pcomtotal = "Pcomtotal" in var["functions"]
    # if has_Pcomtotal:
    #     Pcomtotalx = var["functions"]["Pcomtotal"][0, ii]
    #     Pcomtotaly = var["functions"]["Pcomtotal"][1, ii]
    #     Pcomtotalz = var["functions"]["Pcomtotal"][2, ii]

    # -----------------------------
    # Plot robot skeleton
    # -----------------------------
    ax.plot(Px, Py, Pz, color='k', marker='o', linestyle='-',
            markersize=marker_size, linewidth=line_width)
    
    ax.plot(Px[0:2], Py[0:2], Pz[0:2], color='b', marker='o', linestyle='-',
            markersize=marker_size, linewidth=line_width)  # base link in blue

    # -----------------------------
    # Plot COM markers
    # -----------------------------
    ax.scatter(Pcomx, Pcomy, Pcomz, color='c', s=50)

    # if has_Pcomtotal:
    #     ax.scatter(Pcomtotalx, Pcomtotaly, Pcomtotalz, color=[0.2, 0.7, 0.05],
    #                marker='s', s=70, label='Total COM')

    return np.column_stack((Px, Py, Pz)), None, None, np.column_stack((Pcomx, Pcomy, Pcomz))


def padded_lims(vmin, vmax, pad_frac=0.05):
            rng = vmax - vmin
            if rng == 0:
                pad = max(1e-3, abs(vmin) * pad_frac)
            else:
                pad = rng * pad_frac
            return (vmin - pad, vmax + pad)

def plot_joint_traj_from_vars(num_vars, *args, **kwargs):
    """
    Plots joint trajectories: q, dq, ddq, and model_tau.
    
    Args:
        num_vars (dict): dictionary containing 'variables', 'functions', 'parameters'
        *args, **kwargs: additional plot arguments (e.g., color, linestyle)
    """
    q = num_vars["variables"]["q"]
    dq = num_vars["variables"]["dq"]
    ddq = num_vars["variables"]["ddq"]
    tau = num_vars["functions"]["model_tau"]
    q_min, q_max = np.nanmin((q)), np.nanmax((q))
    q_ylim = padded_lims(q_min, q_max)

    n, Nq = q.shape
    _, Ndq = dq.shape
    _, Nddq = ddq.shape
    _, Ntau = tau.shape

    dt = num_vars["parameters"]["dt"]

    tq = np.linspace(0, (Nq-1)*dt, Nq)
    tdq = np.linspace(0, (Ndq-1)*dt, Ndq)
    tddq = np.linspace(0, (Nddq-1)*dt, Nddq)
    ttau = np.linspace(0, (Ntau-1)*dt, Ntau)

    for ii in range(n):
        # -----------------------
        # q
        plt.subplot(n, 4, ii*4 + 1)
        plt.plot(tq, q[ii, :], *args, **kwargs)
        plt.gca().set_ylim(q_ylim)

        if "qmin" in num_vars["parameters"] and "qmax" in num_vars["parameters"]:
            plt.plot(tq[[0, -1]], [num_vars["parameters"]["qmin"][ii]]*2, color=[0,0.7,0.7], linestyle='--')
            plt.plot(tq[[0, -1]], [num_vars["parameters"]["qmax"][ii]]*2, color=[0.7,0,0.7], linestyle='--')
        plt.ylabel(f"$q_{{{ii+1}}}$")
        plt.grid(True)
        if ii == n-1:
            plt.xlabel("$t$ [s]")

        # -----------------------
        # dq
        plt.subplot(n, 4, ii*4 + 2)
        plt.plot(tdq, dq[ii, :], *args, **kwargs)
        if "dqmin" in num_vars["parameters"] and "dqmax" in num_vars["parameters"]:
            plt.plot(tdq[[0, -1]], [num_vars["parameters"]["dqmin"][ii]]*2, color=[0,0.7,0.7], linestyle='--')
            plt.plot(tdq[[0, -1]], [num_vars["parameters"]["dqmax"][ii]]*2, color=[0.7,0,0.7], linestyle='--')
        plt.ylabel(f"$\\dot{{q}}_{{{ii+1}}}$")
        plt.grid(True)
        if ii == n-1:
            plt.xlabel("$t$ [s]")

        # -----------------------
        # ddq
        plt.subplot(n, 4, ii*4 + 3)
        plt.plot(tddq, ddq[ii, :], *args, **kwargs)
        plt.ylabel(f"$\\ddot{{q}}_{{{ii+1}}}$")
        plt.grid(True)
        if ii == n-1:
            plt.xlabel("$t$ [s]")

        # -----------------------
        # tau
        plt.subplot(n, 4, ii*4 + 4)
        plt.plot(ttau, tau[ii, :], *args, **kwargs)
        if "taumin" in num_vars["parameters"] and "taumax" in num_vars["parameters"]:
            plt.plot(ttau[[0, -1]], [num_vars["parameters"]["taumin"][ii]]*2, color=[0,0.7,0.7], linestyle='--')
            plt.plot(ttau[[0, -1]], [num_vars["parameters"]["taumax"][ii]]*2, color=[0.7,0,0.7], linestyle='--')
        plt.ylabel(f"$\\tau_{{{ii+1}}}$")
        plt.grid(True)
        plt.legend()
        if ii == n-1:
            plt.xlabel("$t$ [s]")
            

def plot_segment_vels_from_vars(vars, *args, **kwargs):
    """
    Plots segment velocities (V) and numerical velocities (Vnum) for each segment.
    
    Args:
        vars (dict): robot variables and functions
        *args, **kwargs: optional plotting arguments (color, linestyle, etc.)
    """
    q = vars["variables"]["q"]
    n, N = q.shape
    dt = vars["parameters"]["dt"]
    t = np.linspace(0, (N-1)*dt, N)

    for ii in range(n):
        # Compute numerical velocities
        P_next = vars["functions"]["P"][ii+1]
        Vnum = np.diff(P_next, axis=1) / dt
        NnumV = Vnum.shape[1]

        # ----------------------
        # x-velocity
        # ----------------------
        # plt.subplot(3, n, ii+1)
        # plt.plot(t, vars["functions"]["V"][ii+1][0, :], *args, **kwargs)
        # plt.plot(t[:NnumV], Vnum[0, :], '--', *args, **kwargs)
        # plt.ylabel(f"$V^{{{ii+1}}}_x$")
        # plt.legend()
        # plt.grid(True)

        # # ----------------------
        # # y-velocity
        # # ----------------------
        # plt.subplot(3, n, n + ii+1)
        # plt.plot(t, vars["functions"]["V"][ii+1][1, :], *args, **kwargs)
        # plt.plot(t[:NnumV], Vnum[1, :], '--', *args, **kwargs)
        # plt.ylabel(f"$V^{{{ii+1}}}_y$")
        # plt.legend()
        # plt.grid(True)

        # ----------------------
        # theta-velocity
        # ----------------------
        # plt.subplot(3, n, 2*n + ii+1)
        # plt.plot(t, vars["functions"]["V"][ii+1][2, :], *args, **kwargs)
        # plt.plot(t[:NnumV], Vnum[2, :], '--', *args, **kwargs)
        # plt.ylabel(f"$V^{{{ii+1}}}_\\theta$")
        # plt.legend()
        # plt.grid(True)

        # xlabel for bottom row
        plt.xlabel("$t$ [s]")