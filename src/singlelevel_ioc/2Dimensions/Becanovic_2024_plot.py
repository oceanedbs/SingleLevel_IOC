import numpy as np
import matplotlib.pyplot as plt

def plot_snapshots_from_vars(var, nsnapshots=10):
    """
    Plots snapshots of the robot in a single figure.
    
    Args:
        vars (dict): Dictionary containing variables and functions, same as in the model.
        nsnapshots (int): Number of snapshots to display along the trajectory.
    """
    q = var["variables"]["q"]
    n, N = q.shape  # number of joints, number of timesteps

    # -----------------------------
    # Plot end-effector trajectory
    # -----------------------------
    P_end = var["functions"]["P"][-1]  # last link EE position (3 x N)
    plt.plot(P_end[0, :], P_end[1, :], color=[1, 1, 0], linewidth=2, label="EE traj")

    # -----------------------------
    # Plot segment COM trajectories
    # -----------------------------
    for ii in range(n):
        Pcom = var["functions"]["Pcom"][ii]
        plt.plot(Pcom[0, :], Pcom[1, :], color=[0, 1, 1], linewidth=2, label=f"COM {ii+1}", alpha=0.1)

    # -----------------------------
    # Plot total COM trajectory (if exists)
    # -----------------------------
    if "Pcomtotal" in var["functions"]:
        Pcomtotal = var["functions"]["Pcomtotal"]
        plt.plot(Pcomtotal[0, :], Pcomtotal[1, :], color=[0.2, 0.7, 0.05], linewidth=2, label="Total COM")

    # -----------------------------
    # Plot robot configurations at snapshots
    # -----------------------------
    snapshot_indices = np.floor(np.linspace(0, N-1, nsnapshots)).astype(int)
 
    for ii in snapshot_indices:
        P, _, _, _ = snapshot_from_vars(var, ii)
        dP = P[-1, :] - P[-2, :]
        alpha = 0.2
        plt.text(
            P[-1, 0] + alpha * dP[0],
            P[-1, 1] + alpha * dP[1],
            f"{ii}",
            fontsize=16,
            fontweight='bold',
            horizontalalignment='center',
            verticalalignment='center'
        )
        snapshot_from_vars(var, ii)  # Draw the robot at this snapshot

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


def snapshot_from_vars(num_vars, ii, marker_size=15, line_width=2):
    """
    Plots a robot snapshot at timestep ii.

    Args:
        num_vars (dict): robot variables and functions
        ii (int): timestep index
        marker_size (int, optional): marker size. Default 15
        line_width (int, optional): line width. Default 2

    Returns (optional):
        PxPy: np.ndarray of end-effector positions [[x1,y1], ...]
        UV: np.ndarray of vectors Ux, Uy
        VV: np.ndarray of vectors Vx, Vy
        PcomXY: np.ndarray of COM positions [[x1,y1], ...]
    """
    q = num_vars["variables"]["q"]
    n = q.shape[0]

    # -----------------------------
    # Extract positions
    # -----------------------------
    P = np.vstack([np.array(Pi) for Pi in num_vars["functions"]["P"]])  # shape (3*(n+1), N)
    Px = P[0::3, ii]
    Py = P[1::3, ii]
    Ptheta = P[2:3, ii]

    L = np.append(num_vars["parameters"]["L"], num_vars["parameters"]["L"][-1])
    # Direction vectors
    Ux = L * np.cos(Ptheta)
    Vx = L * np.sin(Ptheta)
    Uy = L * (-np.sin(Ptheta))
    Vy = L * np.cos(Ptheta)

    # -----------------------------
    # COM positions
    # -----------------------------
    Pcom = np.hstack(num_vars["functions"]["Pcom"])
    Pcomx = Pcom[0::3, ii]
    Pcomy = Pcom[1::3, ii]

    # Optional total COM
    has_Pcomtotal = "Pcomtotal" in num_vars["functions"]
    if has_Pcomtotal:
        Pcomtotalx = num_vars["functions"]["Pcomtotal"][0, ii]
        Pcomtotaly = num_vars["functions"]["Pcomtotal"][1, ii]

    # -----------------------------
    # Plot robot links
    # -----------------------------
    plt.plot(Px, Py, color=[0, 0, 0], linestyle='-', marker='o',
             markersize=marker_size, linewidth=line_width)

    # Quiver vectors
    plt.quiver(Px, Py, Ux, Vx, scale=10, color=[1, 0, 0], width=0.005, linewidth=line_width, alpha=0.2)
    plt.quiver(Px, Py, Uy, Vy, scale=10, color=[0, 1, 0], width=0.005, linewidth=line_width, alpha=0.2)

    # COM positions
    plt.plot(Pcomx, Pcomy, color='c', linestyle='None', marker='o',
             markersize=marker_size, linewidth=line_width)
    plt.legend()

    if has_Pcomtotal:
        plt.plot(Pcomtotalx, Pcomtotaly, color=[0.2, 0.7, 0.05], linestyle='None',
                 marker='s', markersize=marker_size, linewidth=line_width)

    # -----------------------------
    # Optional outputs
    # -----------------------------
    return np.column_stack((Px, Py)), np.column_stack((Ux, Uy)), np.column_stack((Vx, Vy)), np.column_stack((Pcomx, Pcomy))


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
        plt.subplot(3, n, ii+1)
        plt.plot(t, vars["functions"]["V"][ii+1][0, :], *args, **kwargs)
        plt.plot(t[:NnumV], Vnum[0, :], '--', *args, **kwargs)
        plt.ylabel(f"$V^{{{ii+1}}}_x$")
        plt.legend()
        plt.grid(True)

        # ----------------------
        # y-velocity
        # ----------------------
        plt.subplot(3, n, n + ii+1)
        plt.plot(t, vars["functions"]["V"][ii+1][1, :], *args, **kwargs)
        plt.plot(t[:NnumV], Vnum[1, :], '--', *args, **kwargs)
        plt.ylabel(f"$V^{{{ii+1}}}_y$")
        plt.legend()
        plt.grid(True)

        # ----------------------
        # theta-velocity
        # ----------------------
        plt.subplot(3, n, 2*n + ii+1)
        plt.plot(t, vars["functions"]["V"][ii+1][2, :], *args, **kwargs)
        plt.plot(t[:NnumV], Vnum[2, :], '--', *args, **kwargs)
        plt.ylabel(f"$V^{{{ii+1}}}_\\theta$")
        plt.legend()
        plt.grid(True)

        # xlabel for bottom row
        plt.xlabel("$t$ [s]")