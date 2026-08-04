# Step 2: one-joint stochastic optimal open-loop control

This folder adds the stochastic ingredient needed for functional muscle
co-contraction, but deliberately does so in a separate one-joint model before
changing the 4-DOF code.

## Why this intermediate validation matters

The Step-1 model already enforces

```text
RNEA(q, dq, ddq) = muscle_torque(q, dq, activation)
```

but it is deterministic. In an effort-minimizing deterministic problem,
antagonist co-contraction normally has no task benefit. Step 2 introduces
additive torque noise and propagates the mean and covariance of the stochastic
joint state.

The physical state is

```text
x = [joint angle, joint velocity]
```

and its deterministic equivalent contains

```text
mean(x) = [mean angle, mean velocity]
P = [[P_angle, P_cross],
     [P_cross, P_velocity]]
```

For first-order statistical linearization,

```text
mean_dot = f(mean, activation)
P_dot = A P + P A.T + G G.T
A = df/dx evaluated at the mean
```

Muscle excitation remains the open-loop control and activation follows the same
first-order activation dynamics used in Step 1.

## Objective

The objective is

```text
mean-squared muscle activation + variance_weight * predicted state variance
```

plus a strong mean-posture cost and a zero terminal-mean constraint. There is
**no direct co-contraction cost or co-contraction target**. Co-contraction is
allowed to emerge because summed antagonist activation increases stiffness and
reduces the propagation of noise.

This is also why the normalized coactivation penalty from the Step-1 runner is
not used here. A normalized overlap can be close to one even when both muscles
have very small activation; it is useful as a descriptive index but is not a
suitable magnitude-based effort term for this validation.

## Expected result

The runner solves three cases:

1. no noise: effort minimization drives activation toward zero;
2. torque noise with a moderate variance penalty: nonzero symmetric activation;
3. the same noise with a stronger variance penalty: larger co-contraction and
   stiffness, with smaller predicted angle variance.

The final figures check the propagated moments against 500 nonlinear
Euler-Maruyama simulations driven by the optimized open-loop activation. A
dedicated figure overlays the low- and strong-variability-penalty conditions,
showing every simulated angle and angular-velocity trajectory together with
each condition's pointwise sample mean and mean +/- 1 sample SD. The SD is
computed independently at each time node across all 500 trajectories with
``ddof=1``. Both conditions use the same random seed for a paired comparison.
An additional compact A--D figure follows the layout of the reference figure:
low-penalty sample kinematics (A), low-penalty stiffness and antagonist torque
(B), strong-penalty kinematics (C), and strong-penalty mechanics (D).

## Files

- `sooc_1dof_func.py`: mean/covariance dynamics, RK4 defects, objective, initial
  guess generation, and Monte Carlo validation;
- `run_sooc_1dof.py`: solves and compares the three scenarios;
- `sooc_1dof_plot.py`: comparison and Monte Carlo plots.

## Run

```bash
python3 run_sooc_1dof.py
```

Dependencies are NumPy, Matplotlib, CasADi, and IPOPT through CasADi. Pinocchio
is not needed for this isolated validation.

## Success criteria before Step 3

- flexor and extensor activations remain almost equal, so the mean posture stays
  near zero;
- their activation sum increases when the variance weight increases;
- optimized stiffness approaches or exceeds the destabilizing gravitational
  stiffness `m*g*lc`;
- predicted and Monte Carlo angle standard deviations are reasonably close;
- removing noise removes the functional reason for co-contraction.

Once these checks pass, the same covariance propagation can be generalized to
the 4-DOF mechanical state `[q, dq]`, whose covariance has 36 independent
entries.
