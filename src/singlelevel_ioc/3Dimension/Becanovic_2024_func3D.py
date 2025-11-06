import casadi as ca
import numpy as np
from roboticstoolbox import DHRobot
from spatialmath import SE3


def make_ndof_model(n, N, dh_params):
    if N <= 2:
        raise ValueError("Need N > 2.")
    
    opti = ca.Opti()

    var = {}

    # -------------------------------
    # Parameters
    # -------------------------------
    params = {}
    params['dt'] = opti.parameter(1)
    params['q0'] = opti.parameter(n)
    params['dq0'] = opti.parameter(n)
    params['L'] = opti.parameter(n)
    params['COM'] = opti.parameter(3, n)
    params['M'] = opti.parameter(n)
    params['I'] = opti.parameter(n)
    params['gravity'] = opti.parameter(3)
    params['goal'] = opti.parameter(3, 1)

    params['Fext'] = []
    for ii in range(n):
        params['Fext'].append(opti.parameter(3, N - 2))


   
    var['parameters'] = params

    # -------------------------------
    # Variables
    # -------------------------------
    variables = {}
    variables['ddq'] = opti.variable(n, N - 2)
    variables['dq'] = opti.variable(n, N - 1)
    variables['q'] = opti.variable(n, N)
    var['variables'] = variables

    # -------------------------------
    # Functions
    # -------------------------------
    functions = {}
    functions['q'] = variables['q']
    functions['dq'] = ca.horzcat(variables['dq'], ca.DM.zeros(n, 1))
    functions['ddq'] = ca.horzcat(variables['ddq'], ca.DM.zeros(n, 2))

    functions['Fext'] = []
    for ii in range(n):
        functions['Fext'].append(
            ca.horzcat(params['Fext'][ii], ca.DM.zeros(3, 2))
        )
     
    # Call your custom dynamics
    (functions['P'], 
     functions['V'], 
    #functions['A'], 
    functions['Pcom'],
    functions['Vcom'], 
    #functions['Acom'],
    functions['Fcom'], 
    functions['Ncom']) = forward_propagation( 
         functions['q'],
         functions['dq'],
         functions['ddq'],
         dh_params, 
         params['M'], 
         params['COM'], 
         params['I'], 
         params['gravity']
     )


    functions['F'], functions['N'], functions['model_tau'] = backward_propagation(
        functions['Fcom'],
        functions['Ncom'],
        functions['Fext'],
        params['M'],
        params['gravity'],
        functions['q'],
        dh_params
        
    )
   
    var['functions'] = functions

    # -------------------------------
    # Constraints
    # -------------------------------
    constraints = {}
    constraints['initial_pos'] = variables['q'][:, 0] - params['q0']
    constraints['initial_vel'] = variables['dq'][:, 0] - params['dq0']
    constraints['dynamics_pos'] = variables['q'][:, 1:N] - variables['q'][:, 0:N-1] - variables['dq'] * params['dt']
    constraints['dynamics_vel'] = variables['dq'][:, 1:N-1] - variables['dq'][:, 0:N-2] - variables['ddq'] * params['dt']
    constraints['goal_ee'] = functions['P'][-1][0:3, -1] - params['goal']

    var['constraints'] = constraints

    # -------------------------------
    # Costs
    # -------------------------------
    costs = {}
    tau = functions['model_tau']
    dq = variables['dq']
    ddq = variables['ddq']
    Vee = functions['V'][-1]

    costs['joint_torque_cost'] = ca.sumsqr(tau) / N / n / 8e1
    costs['joint_vel_cost'] = ca.sumsqr(dq) / N / n / 3e0
    costs['ee_vel_cost'] = ca.sumsqr(Vee) / N / 2e1
    costs['joint_torque_change_cost'] = ca.sumsqr(tau[:, 1:] - tau[:, :-1]) / (params['dt'] ** 2) / N / n / 6e5
    costs['joint_jerk_cost'] = ca.sumsqr(ddq[:, 1:] - ddq[:, :-1]) / (params['dt'] ** 2) / N / n / 2e6

    var['costs'] = costs

    # -------------------------------
    # Add constraints to opti
    # -------------------------------
    opti.subject_to(constraints['initial_pos'] == 0)
    opti.subject_to(constraints['initial_vel'] == 0)
    opti.subject_to(constraints['dynamics_pos'] == 0)
    opti.subject_to(constraints['dynamics_vel'] == 0)
    opti.subject_to(constraints['goal_ee'] == 0)

    return opti, var

def mdh_transform(a, alpha, d, theta):
    ct = ca.cos(theta)
    st = ca.sin(theta)
    ca_ = ca.cos(alpha)
    sa = ca.sin(alpha)

    return ca.vertcat(
        ca.horzcat(ct, -st, 0, a),
        ca.horzcat(st*ca_, ct*ca_, -sa, -d*sa),
        ca.horzcat(st*sa, ct*sa, ca_, d*ca_),
        ca.horzcat(0,0,0,1)
    )


def forward_propagation(q, dq, ddq, dh_params, M, COM, I, gravity):
    """
    Compute positions, velocities, accelerations of joints in 3D
    using symbolic DH parameters (CasADi MX).

    Inputs:
        q   : n x N joint positions
        dq  : n x N joint velocities
        ddq : n x N joint accelerations
        dh_params : list of n tuples (a, alpha, d) 
                    (link lengths, twist, offset)
        M   : list of link masses
        COM : 3 x n matrix of COM positions in link frame
        I   : list of inertia matrices
        gravity : 3 x 1 gravity vector

    Outputs:
        P     : list of n+1 CasADi MX (3 x N) joint positions
        V     : list of n+1 CasADi MX (3 x N) joint linear velocities
        Pcom  : list of n CasADi MX (3 x N) COM positions
        Vcom  : list of n CasADi MX (3 x N) COM velocities
        Fcom  : list of n CasADi MX (3 x N) forces at COM
        Ncom  : list of n CasADi MX (3 x N) moments at COM
    """

    n, N = q.shape

    print(dh_params)

    # Initialize outputs
    P = [ca.MX.zeros(3, N) for _ in range(n+1)]
    V = [ca.MX.zeros(3, N) for _ in range(n+1)]

    Pcom, Vcom, Fcom, Ncom = [], [], [], []
    z0 = ca.MX([0, 0, 1])

    for i in range(n):
        Pcom.append(ca.MX.zeros(3, N))
        Vcom.append(ca.MX.zeros(3, N))
        Fcom.append(ca.MX.zeros(3, N))
        Ncom.append(ca.MX.zeros(3, N))

    for t in range(N):
        R_prev = ca.MX.eye(3)
        p_prev = ca.MX.zeros(3)
        a_prev = ca.MX.zeros(3)
        a_prev = gravity
        omega_prev = ca.MX.zeros(3)
        domega_i_prev = ca.MX.zeros(3)

        for i in range(n):
            l_i, alpha_i, d_i = dh_params[i]
            theta_i = q[i, t]

            # Compute transform
            T_i_joint = mdh_transform(l_i, alpha_i, d_i, theta_i)
            R_local = T_i_joint[:3, :3]
            p_local = T_i_joint[:3, 3]

            # Angular velocity & acceleration (revolute)
            omega_i = R_local.T @ domega_i_prev + z0 * dq[i, t]
            domega_i = R_local.T @ domega_i_prev + z0 * ddq[i, t] + ca.cross(R_local.T @ omega_prev, z0 * dq[i, t])

            # Linear acceleration
            a_i = a_prev + R_local.T @ ca.cross(domega_i_prev, p_local) + ca.cross(omega_prev, ca.cross(omega_prev, p_local))

            # --- COM acceleration ---
            a_com_i = a_i + ca.cross(domega_i, COM[:, i]) + ca.cross(omega_i, ca.cross(omega_i, COM[:, i]))
           
            # Force at COM
            Fcom[i][:, t] = M[i] * a_com_i

            # Moment at COM
            Ncom[i][:, t] = I[i] @ domega_i + ca.cross(omega_i, I[i] @ omega_i)

            # Update position
            R_i = R_prev @ R_local
            p_i = p_prev + R_i @ p_local
            p_com_i = p_prev + R_i @ COM[:, i]

            # --- Linear velocity of joint i  and com i ---
            v_i = R_local.T @ (V[i][:, t-1] if t > 0 else ca.MX.zeros(3)) \
                  + ca.cross(omega_i, p_local)
            v_com_i = v_i + ca.cross(omega_i, R_i @ COM[:, i])

            # Store results
            P[i+1][:, t] = p_i
            Pcom[i][:, t] = p_com_i
            V[i+1][:, t] = v_i
            Vcom[i][:, t] = v_com_i

            # Update for next link
            R_prev = R_i
            p_prev = p_i
            a_prev = a_i
            omega_prev = omega_i
            domega_i_prev = domega_i

    return P, V, Pcom, Vcom, Fcom, Ncom


def backward_propagation(Fcom, Ncom, Fext, M, gravity, q, dh_params):
    """
    3D version of backward_propagation using CasADi MX.

    Computes the total forces and moments (torques) at each joint
    by recursively propagating from the end-effector toward the base.

    Inputs:
        Fcom    : list of n CasADi MX (3 x N), inertial forces at each link's COM
        Fext    : list of n CasADi MX (3 x N), external forces applied to each link
        Pcom    : list of n CasADi MX (3 x N), COM positions of each link
        P       : list of n+1 CasADi MX (3 x N), joint positions
        M       : (n,) link masses
        gravity : (3,) CasADi MX or DM, gravity vector (e.g., [0, 0, -9.81])

    Outputs:
        F : list of n+1 CasADi MX (3 x N), total forces at each joint
        N : list of n+1 CasADi MX (3 x N), total moments (torques) at each joint
    """

    n = len(Fcom)
    _, Nsteps = Fcom[0].shape
    g = ca.MX(gravity).reshape((3,1))
    z0 = ca.MX([0, 0, 1])

    # Initialize forces
    F = [ca.MX.zeros(3, Nsteps) for _ in range(n)]
    N = [ca.MX.zeros(3, Nsteps) for _ in range(n)]
    tau = ca.MX.zeros(n, Nsteps)

    for t in range(Nsteps):
        f_prev = ca.MX.zeros(3)
        n_prev = ca.MX.zeros(3)

        for i in reversed(range(n)):
            if i == n-1:
                R = ca.MX.eye(3)
                p = ca.MX.zeros(3)
            else:
                l_i, alpha_i, d_i = dh_params[i+1]
                theta_i = q[i+1, t]
                T_joint = mdh_transform(l_i, alpha_i, d_i, theta_i)
                R = T_joint[:3, :3]

            # Inertial force + gravity + external + propagated from child
            Fm = Fcom[i][:, t] + Fext[i][:, t]
            Fi = R @ f_prev + Fm
            Ni = R @ n_prev + ca.cross(p, R @ f_prev) + ca.cross(p, Fm) + Ncom[i][:, t]
            taui = Ni.T @ z0

            # Store total force at this joint
            F[i][:, t] = Fi
            N[i][:, t] = Ni
            tau[i, t] = taui

            # Update propagated force for next iteration
            f_prev = Fi

    return F, N, tau


def instantiate_ndof_model(var, opti, dt, q0, dq0, L, COM, M, I, gravity, Fext, goal, ddq, dq, q):
    """
    Python translation of MATLAB instantiate_ndof_model.
    
    Sets all parameter values and initial guesses in the CasADi Opti instance.

    Inputs:
        var     : dictionary returned by make_ndof_model()
        opti     : casadi.Opti() instance
        dt       : scalar timestep
        q0, dq0  : (n,) initial position and velocity
        L, COM, M, I : robot link parameters
        gravity  : (2,) gravity vector
        Fext     : list of n arrays (3 x N-2) external forces
        goal     : (2,1) target end-effector position
        ddq, dq, q : initial guesses for variables
    """

    n = q0.shape[0]
    print(n)

    # -------------------------------
    # Parameters
    # -------------------------------
    p = var['parameters']
    opti.set_value(p['dt'], dt)
    opti.set_value(p['q0'], q0)
    opti.set_value(p['dq0'], dq0)
    opti.set_value(p['L'], L)
    opti.set_value(p['COM'], COM)
    opti.set_value(p['M'], M)
    opti.set_value(p['I'], I)
    opti.set_value(p['gravity'], gravity)

    for ii in range(n):
        opti.set_value(p['Fext'][ii], Fext[ii])

    opti.set_value(p['goal'], goal)

    # -------------------------------
    # Variables (initial guesses)
    # -------------------------------
    v = var['variables']
    opti.set_initial(v['ddq'], ddq)
    opti.set_initial(v['dq'], dq)
    opti.set_initial(v['q'], q)


def numerize_var(model_var, opti, initial_flag=False):
    """
    Evaluate all symbolic CasADi variables/parameters/functions in a model
    into numeric CasADi DM arrays, either at the current solution or at
    the initial guess.

    Args:
        model_var (dict): The structure returned by make_ndof_model().
        opti (casadi.Opti): The CasADi Opti instance.
        initial_flag (bool, optional): 
            If True, evaluate using opti.initial(). 
            Otherwise, use the optimized values. Default: False.

    Returns:
        dict: A dictionary 'num_var' with the same structure as model_var,
              but all CasADi symbols replaced by numeric DM values.
    """

    num_var = {}

    # Loop over main categories: 'variables', 'parameters', 'functions'
    for category_name, category_content in model_var.items():
        num_var[category_name] = {}

        # Loop over computables within each category
        for computable_name, computable_value in category_content.items():
            # Case 1: list (cell array in MATLAB)
            if isinstance(computable_value, (list, tuple)):
                num_var[category_name][computable_name] = []
                for item in computable_value:
                    if not initial_flag:
                        num_var[category_name][computable_name].append(opti.value(item))
                    else:
                        num_var[category_name][computable_name].append(
                            opti.value(item, opti.initial())
                        )

            # Case 2: direct symbolic expression (MX/DM)
            else:
                if not initial_flag:
                    num_var[category_name][computable_name] = opti.value(computable_value)
                else:
                    num_var[category_name][computable_name] = opti.value(
                        computable_value, opti.initial()
                    )

    return num_var

