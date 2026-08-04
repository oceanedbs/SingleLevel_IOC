# Step 1: deterministic 4-DOF muscle-driven DOC

This folder is a first deterministic implementation in the same CasADi/Pinocchio style as the supplied double-pendulum code.

## Model

The subject-specific fixed-base URDF contains four generalized coordinates in
Pinocchio order:

1. `subtalar_c`: ankle eversion/inversion;
2. `ankle_c`: ankle dorsiflexion/plantarflexion;
3. `hip_c_rotation2`: hip flexion/extension;
4. `hip_c_rotation1`: hip abduction/adduction.

The abstract positive/negative actuators are labeled by their exact URDF
coordinate because the subject-specific axes are oblique and their anatomical
sign must not be inferred from the coordinate name alone.

Eight abstract muscles form four antagonist pairs. Neural excitation is the control and muscle activation is a state:

```text
a[k+1] = a[k] + dt * (e[k] - a[k]) / tau_activation
```

The essential muscle-driven constraint is:

```text
RNEA(q[k], dq[k], ddq[k]) = tau_muscle(q[k], dq[k], a[k])
```

The reduced muscle force law is activation-dependent and viscoelastic. It is designed so that differential activation generates net torque while summed antagonist activation increases joint stiffness and damping.

The objective contains mean-squared muscle activation, joint torque, joint
velocity, joint jerk, COM velocity, COM acceleration, COM jerk, and normalized
antagonist co-contraction terms. Their relative influence is configured by the `weights`
dictionary in `run_muscle_doc.py`.

## Files

- `subject_model.urdf`: subject-specific 4-DOF model used by the runner.
- `four_dof_ankle_hip.urdf`: original simplified 4-DOF test model.
- `muscle_doc_func.py`: model construction, muscle dynamics, constraints and costs.
- `run_muscle_doc.py`: parameter values, initialization, IPOPT solve and diagnostics.
- `muscle_doc_plot.py`: joint, torque, activation, force, stiffness, COM and coactivation plots.

## Run

From this directory, in the environment containing Pinocchio, CasADi, NumPy and Matplotlib:

```bash
python3 run_muscle_doc.py
```

## What this step validates

- muscle excitation and activation are optimization variables;
- muscle torques, rather than unconstrained RNEA torques, actuate the motion;
- RNEA torque and muscle torque match at every collocation interval;
- antagonist activation changes model stiffness;
- a 4-DOF movement reaches prescribed final COM x and y coordinates while COM
  height remains unconstrained.
- the reference final posture is computed by bounded inverse kinematics from a
  prescribed planar COM target `(x, y)`.

This remains deterministic. With an activation-effort objective, meaningful co-contraction is not expected to emerge by itself. The later stochastic mean/covariance formulation supplies the functional reason for co-contraction.
