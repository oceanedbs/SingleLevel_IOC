import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from Becanovic_2024_func3D import make_ndof_model, instantiate_ndof_model, numerize_var
import sympy as sp
from Becanovic_2024_plot3D import *
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
from Becanovic_2024_Arm import Arm4dof as Arm
import time

n = 2  # number of joints
N = 50  # number of time steps

T = 1.5 # total time
dt = T/(N-1)  # time step

q0 = np.array([np.pi/2, 0])  # initial position
dq0 = np.zeros(n)  # initial velocity

L = np.array([1, 1])  # segment lengths
COM = np.vstack((L/2, np.zeros((1, n)), np.zeros((1, n))))
M = np.array([1, 1])  # segment masses
I = np.array([1/12, 1/12])  # segment inertias
alpha = np.zeros(n)
offset=np.zeros(n)

gravity = np.array([  -9.81, 0, 0])  # gravity vector
Fext = [np.zeros((3, N - 2)) for _ in range(n)]

goal = np.array([[0.5], [0.5], [0]])

A = Arm(L, M, I, COM, alpha, offset, n)
arm = A.create_DH_model()

dh_param = A.get_dh_params()


goal = arm.fkine(np.array([1.99, -2.41, 0]))

# extract last column (position) from homogeneous transform returned by fkine
gmat = goal.A if hasattr(goal, "A") else np.array(goal)
goal = gmat[:3, 3].reshape(3, 1)
print(goal)

#arm.plot(np.array([0,0,0]), block=True)
# arm.plot(np.array([np.pi/2,np.pi/4, 0]), block=True)


# Initial guess
q = np.array((np.linspace(1.5708, 2.0399, N), np.linspace(0, -2.2524, N)))  # initial guess
dq = np.diff(q, axis=1) / dt
ddq = np.diff(dq, axis=1) / dt 

opti, var = make_ndof_model(n, N, dh_param) 


# Instanciate 
instantiate_ndof_model(var, opti, dt, q0, dq0, L, COM, M, I, gravity, Fext, goal, ddq, dq, q);

opti.solver('ipopt')#, {
#     'ipopt.print_level': 5,
#     'ipopt.max_iter': 500,
#     'ipopt.tol': 1e-6,
#     'ipopt.acceptable_tol': 1e-5,
#     'ipopt.acceptable_constr_viol_tol': 1e-3,
#     'ipopt.constr_viol_tol': 1e-4,
#     'ipopt.acceptable_obj_change_tol': 1e-4
# }


# # Optimize joint velocity
theta_1 = np.array([1, 5000, 1])  # weights for cost function 1
theta_1 = theta_1 / np.linalg.norm(theta_1)



opti.minimize(theta_1[0]* var['costs']['joint_vel_cost'] + theta_1[1]* var['costs']['joint_torque_cost']+ theta_1[2]* var['costs']['ee_vel_cost'])

try:
    sol_1 = opti.solve()
except Exception as e:
    print("Solver failed:", e)

    constr_values = opti.debug.value(opti.g)  # returns a numpy array
    print("Constraint values:", constr_values)
    print("Max constraint violation:", np.max(np.abs(constr_values)))

    constr_values = opti.debug.value(opti.g)
    violated = np.where(np.abs(constr_values) > 1e-3)[0]  # threshold can be adjusted
    for idx in violated:
        print(f"Constraint {idx} violated by {constr_values[idx]}")



lambda_1 = sol_1.value(opti.lam_g) # Extract dual variables

# Extract primal variables
q_1 = sol_1.value(var['variables']['q'])
dq_1 = sol_1.value(var['variables']['dq'])
ddq_1 = sol_1.value(var['variables']['ddq'])

# Numerize
num_vars_1 = numerize_var(var, sol_1)


## Make IOC
[opti_ioc, vars_ioc] = make_ndof_model(n, N, dh_param)

# Extract dual variables size
ndual = len(lambda_1)
# Extract parameters size
nparam = len(theta_1)


# Create dual variable parameter
vars_ioc["variables"]["lambda"] = opti_ioc.variable(ndual)
# Create model parameter
vars_ioc["variables"]["theta"] = opti_ioc.variable(nparam)


# Prepare stationarity constraint
vars_ioc["costs"]["compound_cost"] = vars_ioc["variables"]["theta"][0] * vars_ioc["costs"]["joint_vel_cost"] + vars_ioc["variables"]["theta"][1] * vars_ioc["costs"]["joint_torque_cost"] + vars_ioc["variables"]["theta"][2] * vars_ioc["costs"]["ee_vel_cost"]
q_vec = ca.vec(vars_ioc["variables"]["q"])
dq_vec = ca.vec(vars_ioc["variables"]["dq"])
ddq_vec = ca.vec(vars_ioc["variables"]["ddq"])

all_vars = ca.vertcat(q_vec, dq_vec, ddq_vec)

# -----------------------------
# Compute gradient of compound cost w.r.t all variables
# -----------------------------
grad_compound_cost = ca.jacobian(vars_ioc["costs"]["compound_cost"], all_vars).T  # transpose to match MATLAB
vars_ioc["costs"]["grad_compound_cost"] = grad_compound_cost


init_pos = ca.vec(vars_ioc["constraints"]["initial_pos"]) 
init_vel = ca.vec(vars_ioc["constraints"]["initial_vel"])
dynamics_pos = ca.vec(vars_ioc["constraints"]["dynamics_pos"])
dynamics_vel = ca.vec(vars_ioc["constraints"]["dynamics_vel"])
goal_ee = ca.vec(vars_ioc["constraints"]["goal_ee"])

# Concatenate all constraints
compound_constraints = ca.vertcat(init_pos, init_vel, dynamics_pos, dynamics_vel, goal_ee)
vars_ioc["constraints"]["compound_constraints"] = compound_constraints

# Compute gradient of compound constraints w.r.t all variables (use the original all_vars)
vars_ioc["constraints"]["grad_compound_constraints"] = ca.jacobian(compound_constraints, all_vars)

vars_ioc["constraints"]["stationarity"] = vars_ioc["costs"]["grad_compound_cost"] + vars_ioc["constraints"]["grad_compound_constraints"].T @ vars_ioc["variables"]["lambda"]


# 1. Stationarity constraint
opti_ioc.subject_to(vars_ioc["constraints"]["stationarity"] == 0)

# 2. Theta sum equals 1
opti_ioc.subject_to(ca.sum1(vars_ioc["variables"]["theta"]) == 1)

# 3. Theta non-negativity
opti_ioc.subject_to(vars_ioc["variables"]["theta"] >= 0)

# 4. Joint limits around q_1
q = vars_ioc["variables"]["q"]

# Flatten q - q_1
q_diff = ca.vec(q - q_1)
opti_ioc.subject_to(q_diff <= np.pi / 2)
opti_ioc.subject_to(-q_diff <= np.pi / 2)

# -----------------------------
# Create L2 loss
# -----------------------------
# MATLAB: vars_ioc.costs.L2_loss = sum(sum((vars_ioc.variables.q - q_1).^2));
q_diff = vars_ioc["variables"]["q"] - q_1
vars_ioc["costs"]["L2_loss"] = ca.sumsqr(q_diff)  # sumsqr does exactly sum(sum(...^2))

# -----------------------------
# Minimize
# -----------------------------
opti_ioc.minimize(vars_ioc["costs"]["L2_loss"])

# -----------------------------
# Instantiate model (set parameter values and initial guesses)
# -----------------------------
instantiate_ndof_model(
    var=vars_ioc,
    opti=opti_ioc,
    dt=dt,
    q0=q0,
    dq0=dq0,
    L=L,
    COM=COM,
    M=M,
    I=I,
    gravity=gravity,
    Fext=Fext,
    goal=goal,
    ddq=ddq_1,
    dq=dq_1,
    q=q_1
)

# -----------------------------
# Set initial guesses for duals / theta
# -----------------------------
opti_ioc.set_initial(vars_ioc["variables"]["lambda"], lambda_1)
opti_ioc.set_initial(vars_ioc["variables"]["theta"], theta_1)

# -----------------------------
# Solver options and solve
# -----------------------------
opti_ioc.solver('ipopt')
sol_ioc = opti_ioc.solve()

# -----------------------------
# Numerize
# -----------------------------
num_vars_ioc = numerize_var(vars_ioc, opti_ioc, initial_flag=False)

# -----------------------------
# Print identification
# -----------------------------
theta_true = np.array(theta_1).flatten()
theta_id = np.array(num_vars_ioc["variables"]["theta"]).flatten()

print_str = "True theta = [" + ", ".join(["%.4f" % t for t in theta_true]) + "]"
print_str += "\nId.  theta = [" + ", ".join(["%.4f" % t for t in theta_id]) + "].\n"
print(print_str)

# qtemp = num_vars_ioc["variables"]["q"][:, -1]
# q0 = num_vars_ioc["variables"]["q"][:, 0]
# print(num_vars_ioc["variables"]["q"])
# print(qtemp)
# arm.plot(np.array([q0[0], q0[1], 0]), block=True)

# arm.plot(np.array([qtemp[0], qtemp[1], 0]), block=True)

# -----------------------------
# Snapshots plots
# -----------------------------
# fig = plt.figure(figsize=(12, 8))
# ax = plt.axes(projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_aspect('equal')
# ax.view_init(elev=30, azim=-60)
# plot_snapshots_from_vars(fig, num_vars_1, 5)
# # plot 3D goal point on the 3D axes
# ax.scatter(float(goal[0]), float(goal[1]), float(goal[2]), c='r', s=200, marker='o', depthshade=True, label='Goal')
# plt.legend()
# plt.axis('equal')
# plt.title("Snapshots: num_vars_1")

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')
ax.view_init(elev=30, azim=-60)
plot_snapshots_from_vars(fig, num_vars_ioc, 5)
ax = plt.gca()   # now refers to the 3D axes created inside the function
ax.scatter(goal[0], goal[1], goal[2], c='r', s=200, marker='o', depthshade=True, label='Goal')
plt.legend()
plt.axis('equal')
plt.title("Snapshots: num_vars_ioc")

# -----------------------------
# Joint trajectories
# -----------------------------
plt.figure(figsize=(12, 8))
plot_joint_traj_from_vars(num_vars_1)
plot_joint_traj_from_vars(num_vars_ioc)
plt.title("Joint Trajectories")

# -----------------------------
# Segment velocities
# -----------------------------
# plt.figure(figsize=(12, 8))
# plot_segment_vels_from_vars(num_vars_1)
# plot_segment_vels_from_vars(num_vars_ioc)
# plt.title("Segment Velocities")

plt.show()
