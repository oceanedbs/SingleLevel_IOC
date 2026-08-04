# Step 3: two-DOF ankle stochastic optimal open-loop control

This step generalizes the validated one-joint SOOC formulation to a coupled
2-DOF ankle model while retaining a small enough covariance state for debugging.
It is the bridge between the isolated Step-2 validation and the full 4-DOF
ankle--hip model.

## Mechanical model

The supplied `four_dof_ankle_hip.urdf` is loaded with Pinocchio and its two hip
coordinates are locked. The reduced model contains, in this order:

1. ankle flexion/extension, controlling mainly anterior--posterior COM motion;
2. ankle eversion/inversion, controlling mainly medio--lateral COM motion.

The stochastic mechanical state is

```text
x = [q_flexion, q_eversion, dq_flexion, dq_eversion]
```

and the nonlinear mean dynamics are evaluated with Pinocchio ABA:

```text
mean_dot = f(mean, activation)
```

Four abstract muscles form two antagonist pairs. Excitation is the deterministic
open-loop control and activation follows first-order dynamics. The same reduced
activation-dependent force, stiffness and damping law used in Step 1 is retained.

## Covariance propagation

The 4 x 4 mechanical covariance has ten independent entries. First-order
statistical linearization gives

```text
P_dot = A P + P A.T + G G.T
A = df/dx evaluated at the mean trajectory
```

Torque noise is specified independently for the two ankle coordinates. It is
mapped into acceleration noise through the configuration-dependent mass matrix:

```text
G = [[0],
     [M(q)^(-1) diag(sigma_tau)]]
```

The mean, ten covariance entries and four deterministic activations are
integrated together with RK4 inside the direct-collocation constraints.

## Objective

The objective contains:

- mean-squared muscle activation;
- a strong mean-COM and mean-velocity posture cost;
- predicted COM-position variance in ML and AP directions;
- a small joint-velocity variance term;
- a small excitation-rate regularizer.

There is no direct co-contraction target or co-contraction reward. Co-contraction
is useful only because summed antagonist activation changes stiffness and
therefore changes covariance propagation.

The final activation is constrained to equal the preceding node. This removes
the isolated final-node deactivation artifact observed in the finite-horizon
Step-2 result.

## Default experiments

`run_sooc_2dof_ankle.py` solves five conditions:

1. no noise;
2. sagittal ankle torque noise only;
3. frontal ankle torque noise only;
4. two-axis noise with a low variance penalty;
5. two-axis noise with a strong variance penalty.

The directional cases test whether co-contraction is selected mainly in the
antagonist pair relevant to the disturbed direction. The two-axis conditions
test whether a stronger variability penalty increases both pair activation sums
and reduces predicted COM variability.

The strong two-axis solution is validated with nonlinear Euler--Maruyama Monte
Carlo simulation using Pinocchio ABA and the optimized deterministic activation
trajectory.

## Files

- `sooc_2dof_ankle_func.py`: Pinocchio/CasADi dynamics, muscle model, covariance
  packing, RK4 defects, objective, initialization and Monte Carlo simulation;
- `run_sooc_2dof_ankle.py`: reduced-model construction, five optimization
  scenarios, diagnostics and Monte Carlo validation;
- `sooc_2dof_ankle_plot.py`: time-series, directional-selectivity and sampled
  versus predicted variability plots;
- `four_dof_ankle_hip.urdf`: the simple four-coordinate test model from which
  the two ankle coordinates are retained.

## Run

From this folder, in the environment containing Pinocchio, CasADi, IPOPT, NumPy
and Matplotlib:

```bash
python3 run_sooc_2dof_ankle.py
```

The five scenarios can take longer than Step 2. During initial debugging, keep
only the no-noise, sagittal-noise and strong two-axis cases in
`scenario_definitions`, or reduce `N` from 81 to 61.

## Step-3 success criteria

Before moving to the full 4-DOF ankle--hip covariance model, check that:

- no noise drives activation toward the minimum needed by the mean posture;
- sagittal noise increases the flexion-pair activation sum more than the
  eversion-pair sum;
- frontal noise produces the opposite directional preference;
- two-axis noise activates both pairs;
- increasing the variance weight increases pair activation sums and stiffness;
- predicted COM and joint standard deviations decrease with the stronger
  variance penalty;
- propagated and Monte Carlo standard deviations remain reasonably close;
- the printed covariance eigenvalue is not meaningfully negative;
- the mean posture remains close to the upright reference.

The muscle parameters and noise magnitudes are proof-of-concept values. They
should be calibrated before physiological interpretation or transfer to the
subject-specific URDF.
