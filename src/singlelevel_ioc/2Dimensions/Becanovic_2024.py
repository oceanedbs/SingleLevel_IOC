import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from Becanovic_2024_func import make_ndof_model, instantiate_ndof_model, numerize_var
from Becanovic_2024_plot import plot_snapshots_from_vars, plot_snapshots_from_vars, plot_joint_traj_from_vars, plot_segment_vels_from_vars


n = 2  # number of joints
N = 50  # number of time steps

T = 1.5 # total time
dt = T/(N-1)  # time step

q0 = np.zeros(n)  # initial position
q0[0] = np.pi/2 
dq0 = np.zeros(n)  # initial velocity

L = np.ones(n)  # segment lengths
COM = np.vstack((0.5 * np.ones((1, n)), np.zeros((1, n))))
M = np.ones(n)  # segment masses
I = 1/12 * np.ones(n)  # segment inertias

gravity = np.array([0, -9.81])  # gravity vector
Fext = [np.zeros((3, N - 2)) for _ in range(n)]
    
goal = np.array([[0.5], [0.5]])  # goal position for the end-effector

q = np.array((np.linspace(1.5708, 2.0399, N), np.linspace(0, -2.2524, N)))  # initial guess
dq = np.diff(q, axis=1) / dt
ddq = np.diff(dq, axis=1) / dt 

opti, var = make_ndof_model(n, N) 

# Instanciate 
instantiate_ndof_model(var, opti, dt, q0, dq0, L, COM, M, I, gravity, Fext, goal, ddq, dq, q);


opti.solver('ipopt')


# # Optimize joint velocity
theta_1 = np.array([1, 5000,1,0,0,0,0,0,0,0,0])  # weights for cost function 1
theta_1 = theta_1 / np.linalg.norm(theta_1)

opti.minimize(theta_1[0]* var['costs']['joint_vel_cost'] + theta_1[1]* var['costs']['joint_torque_cost'] + theta_1[2]* var['costs']['ee_vel_cost']) #+ theta_1[3]* var['costs']['joint_torque_change_cost'] + theta_1[4]* var['costs']['joint_jerk_cost'] + theta_1[5]* var['costs']['torque change_cost'] + theta_1[6]* var['costs']['acceleration_cost'] + theta_1[7]* var['costs']['mechanical_work_cost'] + theta_1[8]* var['costs']['duration_cost'] + theta_1[9]* var['costs']['accuracy_cost'] + theta_1[10]* var['costs']['posture_cost'])
sol_1 = opti.solve()
lambda_1 = sol_1.value(opti.lam_g) # Extract dual variables


# Extract primal variables
q_1 = sol_1.value(var['variables']['q'])
dq_1 = sol_1.value(var['variables']['dq'])
ddq_1 = sol_1.value(var['variables']['ddq'])

# Numerize
num_vars_1 = numerize_var(var, sol_1)

print(q_1)

# # theta_2 = np.array([200, 150, 30])  # weights for cost function 2
# # theta_2 = theta_2 / np.linalg.norm(theta_2)
# # opti.minimize(theta_2[0] * num_vars_1["costs"]["joint_vel_cost"] + theta_2[1] * num_vars_1["costs"]["joint_torque_cost"] + theta_2[2] * num_vars_1["costs"]["ee_vel_cost"])
# # sol_2 = opti.solve()
# # lambda_2 = sol_2.value(opti.lam_g) #  Extract dual variables
# # # Extract primal variables
# # q_2 = sol_2.value(var['variables']['q'])
# # dq_2 = sol_2.value(var['variables']['dq'])
# # ddq_2 = sol_2.value(var['variables']['ddq'])

# # # Numerize
# # num_vars_2 = numerize_var(var, sol_2)

# ## Make IOC
# [opti_ioc, vars_ioc] = make_ndof_model(n, N);

# # Extract dual variables size
# ndual = len(lambda_1)
# # Extract parameters size
# nparam = len(theta_1)


# # Create dual variable parameter
# vars_ioc["variables"]["lambda"] = opti_ioc.variable(ndual)
# # Create model parameter
# vars_ioc["variables"]["theta"] = opti_ioc.variable(nparam)

# # Prepare stationarity constraint
# vars_ioc["costs"]["compound_cost"] = vars_ioc["variables"]["theta"][0] * vars_ioc["costs"]["joint_vel_cost"] + vars_ioc["variables"]["theta"][1] * vars_ioc["costs"]["joint_torque_cost"] + vars_ioc["variables"]["theta"][2] * vars_ioc["costs"]["ee_vel_cost"] + vars_ioc["variables"]["theta"][3] * vars_ioc["costs"]["joint_torque_change_cost"] + vars_ioc["variables"]["theta"][4] * vars_ioc["costs"]["joint_jerk_cost"] + vars_ioc["variables"]["theta"][5] * vars_ioc["costs"]["torque change_cost"] + vars_ioc["variables"]["theta"][6] * vars_ioc["costs"]["acceleration_cost"] + vars_ioc["variables"]["theta"][7] * vars_ioc["costs"]["mechanical_work_cost"] + vars_ioc["variables"]["theta"][8] * vars_ioc["costs"]["duration_cost"] + vars_ioc["variables"]["theta"][9] * vars_ioc["costs"]["accuracy_cost"] + vars_ioc["variables"]["theta"][10] * vars_ioc["costs"]["posture_cost"]
# q_vec = ca.vec(vars_ioc["variables"]["q"])
# dq_vec = ca.vec(vars_ioc["variables"]["dq"])
# ddq_vec = ca.vec(vars_ioc["variables"]["ddq"])

# all_vars = ca.vertcat(q_vec, dq_vec, ddq_vec)

# # -----------------------------
# # Compute gradient of compound cost w.r.t all variables
# # -----------------------------
# grad_compound_cost = ca.jacobian(vars_ioc["costs"]["compound_cost"], all_vars).T  # transpose to match MATLAB
# vars_ioc["costs"]["grad_compound_cost"] = grad_compound_cost


# init_pos = ca.vec(vars_ioc["constraints"]["initial_pos"]) 
# init_vel = ca.vec(vars_ioc["constraints"]["initial_vel"])
# dynamics_pos = ca.vec(vars_ioc["constraints"]["dynamics_pos"])
# dynamics_vel = ca.vec(vars_ioc["constraints"]["dynamics_vel"])
# goal_ee = ca.vec(vars_ioc["constraints"]["goal_ee"])

# # Concatenate all constraints
# compound_constraints = ca.vertcat(init_pos, init_vel, dynamics_pos, dynamics_vel, goal_ee)

# vars_ioc["constraints"]["compound_constraints"] = compound_constraints

# # Compute gradient of compound constraints w.r.t all variables (use the original all_vars)
# vars_ioc["constraints"]["grad_compound_constraints"] = ca.jacobian(compound_constraints, all_vars)

# vars_ioc["constraints"]["stationarity"] = vars_ioc["costs"]["grad_compound_cost"] + vars_ioc["constraints"]["grad_compound_constraints"].T @ vars_ioc["variables"]["lambda"]


# # 1. Stationarity constraint
# opti_ioc.subject_to(vars_ioc["constraints"]["stationarity"] == 0)

# # 2. Theta sum equals 1
# opti_ioc.subject_to(ca.sum1(vars_ioc["variables"]["theta"]) == 1)

# # 3. Theta non-negativity
# opti_ioc.subject_to(vars_ioc["variables"]["theta"] >= 0)

# # 4. Joint limits around q_1
# q = vars_ioc["variables"]["q"]

# # Flatten q - q_1
# q_diff = ca.vec(q - q_1)
# opti_ioc.subject_to(q_diff <= np.pi / 4)
# opti_ioc.subject_to(-q_diff <= np.pi / 4)

# # -----------------------------
# # Create L2 loss
# # -----------------------------
# # MATLAB: vars_ioc.costs.L2_loss = sum(sum((vars_ioc.variables.q - q_1).^2));
# q_diff = vars_ioc["variables"]["q"] - q_1
# vars_ioc["costs"]["L2_loss"] = ca.sumsqr(q_diff)  # sumsqr does exactly sum(sum(...^2))

# # -----------------------------
# # Minimize
# # -----------------------------
# opti_ioc.minimize(vars_ioc["costs"]["L2_loss"])

# # -----------------------------
# # Instantiate model (set parameter values and initial guesses)
# # -----------------------------
# instantiate_ndof_model(
#     var=vars_ioc,
#     opti=opti_ioc,
#     dt=dt,
#     q0=q0,
#     dq0=dq0,
#     L=L,
#     COM=COM,
#     M=M,
#     I=I,
#     gravity=gravity,
#     Fext=Fext,
#     goal=goal,
#     ddq=ddq_1,
#     dq=dq_1,
#     q=q_1
# )

# # -----------------------------
# # Set initial guesses for duals / theta
# # -----------------------------
# opti_ioc.set_initial(vars_ioc["variables"]["lambda"], lambda_1)
# opti_ioc.set_initial(vars_ioc["variables"]["theta"], theta_1)

# # -----------------------------
# # Solver options and solve
# # -----------------------------
# opti_ioc.solver('ipopt')
# sol_ioc = opti_ioc.solve()

# # -----------------------------
# # Numerize
# # -----------------------------
# num_vars_ioc = numerize_var(vars_ioc, opti_ioc, initial_flag=False)

# # -----------------------------
# # Print identification
# # -----------------------------
# theta_true = np.array(theta_1).flatten()
# theta_id = np.array(num_vars_ioc["variables"]["theta"]).flatten()

# print_str = "True theta = [" + ", ".join(["%.4f" % t for t in theta_true]) + "]"
# print_str += "\nId.  theta = [" + ", ".join(["%.4f" % t for t in theta_id]) + "].\n"
# print(print_str)

# # -----------------------------
# # Snapshots plots
# # -----------------------------
# plt.figure(figsize=(12, 8))

# plot_snapshots_from_vars(num_vars_1, 10)
# plt.plot(goal[0], goal[1], 'ro', markerfacecolor='auto', markersize=20)
# plt.axis('equal')
# plt.title("Snapshots: num_vars_1")

# # plt.figure(figsize=(12, 8))
# # plot_snapshots_from_vars(num_vars_2, 10)
# # plt.plot(goal[0], goal[1], 'ro', markerfacecolor='auto', markersize=20)
# # plt.axis('equal')
# # plt.title("Snapshots: num_vars_2")

# plt.figure(figsize=(12, 8))
# plot_snapshots_from_vars(num_vars_ioc, 10)
# plt.plot(goal[0], goal[1], 'ro', markerfacecolor='auto', markersize=20)
# plt.axis('equal')
# plt.title("Snapshots: num_vars_ioc")

# # -----------------------------
# # Joint trajectories
# # -----------------------------
# plt.figure(figsize=(12, 8))
# plot_joint_traj_from_vars(num_vars_1)
# # plot_joint_traj_from_vars(num_vars_2)
# plot_joint_traj_from_vars(num_vars_ioc)
# plt.title("Joint Trajectories")

# # -----------------------------
# # Segment velocities
# # -----------------------------
# plt.figure(figsize=(12, 8))
# plot_segment_vels_from_vars(num_vars_1)
# # plot_segment_vels_from_vars(num_vars_2)
# plot_segment_vels_from_vars(num_vars_ioc)
# plt.title("Segment Velocities")

# plt.show()