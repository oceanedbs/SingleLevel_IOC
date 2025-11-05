import casadi as ca

def make_ndof_model(n, N):
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
    params['COM'] = opti.parameter(2, n)
    params['M'] = opti.parameter(n)
    params['I'] = opti.parameter(n)
    params['gravity'] = opti.parameter(2)
    params['goal'] = opti.parameter(2, 1)

    print(params)

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
    (functions['Pcom'], functions['P'], 
     functions['Vcom'], functions['V'], 
     functions['Acom'], functions['A'], 
     functions['Fcom']) = forward_propagation(
         functions['q'],
         functions['dq'],
         functions['ddq'],
         params['L'],
         params['COM'],
         params['M'],
         params['I']
     )

    functions['F'] = backward_propagation(
        functions['Fcom'],
        functions['Fext'],
        functions['Pcom'],
        functions['P'],
        params['M'],
        params['gravity']
    )

    # -------------------------------
    # Compute model torques
    # -------------------------------
    functions['model_tau'] = ca.MX.zeros(n, N)
    for ii in range(n):
        functions['model_tau'][ii, :] = functions['F'][ii][2, :]

    var['functions'] = functions

    # -------------------------------
    # Constraints
    # -------------------------------
    constraints = {}
    constraints['initial_pos'] = variables['q'][:, 0] - params['q0']
    constraints['initial_vel'] = variables['dq'][:, 0] - params['dq0']
    constraints['dynamics_pos'] = variables['q'][:, 1:] - variables['q'][:, :-1] - variables['dq'] * params['dt']
    constraints['dynamics_vel'] = variables['dq'][:, 1:] - variables['dq'][:, :-1] - variables['ddq'] * params['dt']
    constraints['goal_ee'] = functions['P'][-1][0:2, -1] - params['goal']
    
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
    # costs['torque change_cost'] = ca.sumsqr(tau[:, 2:] - 2 * tau[:, 1:-1] + tau[:, :-2]) / (params['dt'] ** 4) / N / n / 1e7
    # costs['acceleration_cost'] = ca.sumsqr(ddq) / N / n / 1e1
    # costs['mechanical_work_cost'] = ca.sumsqr(tau[:, 1:] * dq[:, :]) / N / (n - 1) / 1e2
    # costs['duration_cost'] = params['dt'] * N / 1e1
    # costs['accuracy_cost'] = ca.sumsqr(constraints['goal_ee']) * 1e3
    # costs['posture_cost'] = ca.sumsqr(variables['q']) / N / n / 1e2
    # How to add coordination cost 


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



def forward_propagation(q, dq, ddq, L, COM, M, I):
    """
    Python translation of the MATLAB forward_propagation function.
    
    Inputs:
        q   : n x N joint positions
        dq  : n x N joint velocities
        ddq : n x N joint accelerations
        L   : (n,) link lengths
        COM : 2 x n  COM position for each link in local coordinates
        M   : (n,) link masses
        I   : (n,) link inertias

    Outputs:
        Pcom, P, Vcom, V, Acom, A, Fcom : lists of CasADi MX or DM
    """

    n = q.shape[0]
    N = q.shape[1]

    # Initialization
    P = [ca.DM.zeros(3, N)]
    V = [ca.DM.zeros(3, N)]
    A = [ca.DM.zeros(3, N)]

    Pcom = []
    Vcom = []
    Acom = []
    Fcom = []

    for k in range(n):
   
        # --- Precompute reused quantities ---
        q_1k = q[:k+1, :]
        sumq_1k = ca.sum1(q_1k)  # sum over rows
        cq_1k = ca.cos(sumq_1k)
        sq_1k = ca.sin(sumq_1k)

        dq_1k = dq[:k+1, :]
        sumdq_1k = ca.sum1(dq_1k)
        sqsumdq_1k = sumdq_1k ** 2

        ddq_1k = ddq[:k+1, :]
        sumddq_1k = ca.sum1(ddq_1k)

        # --- Propagate positions ---
        Pcom_k = ca.MX.zeros(3, N)
        Pcom_k[0, :] = P[k][0, :] + COM[0, k] * cq_1k - COM[1, k] * sq_1k
        Pcom_k[1, :] = P[k][1, :] + COM[0, k] * sq_1k + COM[1, k] * cq_1k
        Pcom_k[2, :] = P[k][2, :] + q[k, :]

        P_next = ca.MX.zeros(3, N)
        P_next[0, :] = P[k][0, :] + L[k] * cq_1k
        P_next[1, :] = P[k][1, :] + L[k] * sq_1k
        P_next[2, :] = P[k][2, :] + q[k, :]

        # --- Propagate velocities ---
        Vcom_k = ca.MX.zeros(3, N)
        Vcom_k[0, :] = V[k][0, :] + (-COM[0, k] * sq_1k - COM[1, k] * cq_1k) * sumdq_1k
        Vcom_k[1, :] = V[k][1, :] + ( COM[0, k] * cq_1k - COM[1, k] * sq_1k) * sumdq_1k
        Vcom_k[2, :] = V[k][2, :] + dq[k, :]

        V_next = ca.MX.zeros(3, N)
        V_next[0, :] = V[k][0, :] - L[k] * sq_1k * sumdq_1k
        V_next[1, :] = V[k][1, :] + L[k] * cq_1k * sumdq_1k
        V_next[2, :] = V[k][2, :] + dq[k, :]

        # --- Propagate accelerations ---
        Acom_k = ca.MX.zeros(3, N)
        Acom_k[0, :] = (A[k][0, :] +
                        (-COM[0, k] * cq_1k + COM[1, k] * sq_1k) * sqsumdq_1k +
                        (-COM[0, k] * sq_1k - COM[1, k] * cq_1k) * sumddq_1k)
        Acom_k[1, :] = (A[k][1, :] +
                        (-COM[0, k] * sq_1k - COM[1, k] * cq_1k) * sqsumdq_1k +
                        ( COM[0, k] * cq_1k - COM[1, k] * sq_1k) * sumddq_1k)
        Acom_k[2, :] = A[k][2, :] + ddq[k, :]

        A_next = ca.MX.zeros(3, N)
        A_next[0, :] = A[k][0, :] - L[k] * cq_1k * sqsumdq_1k - L[k] * sq_1k * sumddq_1k
        A_next[1, :] = A[k][1, :] - L[k] * sq_1k * sqsumdq_1k + L[k] * cq_1k * sumddq_1k
        A_next[2, :] = A[k][2, :] + ddq[k, :]

        # --- Forces at COM ---
        Fcom_k = ca.MX.zeros(3, N)
        Fcom_k[0, :] = M[k] * Acom_k[0, :]
        Fcom_k[1, :] = M[k] * Acom_k[1, :]
        Fcom_k[2, :] = I[k] * Acom_k[2, :]
        
        Pcom.append(Pcom_k)
        P.append(P_next)
        Vcom.append(Vcom_k)
        V.append(V_next)
        Acom.append(Acom_k)
        A.append(A_next)
        Fcom.append(Fcom_k)

        print('Pcom ', Pcom)
        print('P ', P)
        print('Vcom ', Vcom)
        print('V ', V)
        print('Acom ', Acom)
        print('A ', A)
        print('Fcom ', Fcom)

    return Pcom, P, Vcom, V, Acom, A, Fcom


def backward_propagation(Fcom, Fext, Pcom, P, M, gravity):
    """
    Python translation of MATLAB backward_propagation.
    Uses CasADi MX for symbolic computation.

    Inputs:
        Fcom    : list of n CasADi MX (3 x N) link COM forces
        Fext    : list of n CasADi MX (3 x N) external forces
        Pcom    : list of n CasADi MX (3 x N) COM positions
        P       : list of n+1 CasADi MX (3 x N) joint positions
        M       : (n,) CasADi MX or DM, masses
        gravity : (2,) CasADi MX or DM, gravity vector (x, y)
    
    Output:
        F       : list of n+1 CasADi MX (3 x N) total propagated forces
    """

    n = len(Fcom)
    N = Fcom[-1].shape[1]

    # Initialize force containers
    F = [ca.MX.zeros(3, N) for _ in range(n + 1)]

    # Backward recursion
    for ii in range(n - 1, -1, -1):
        Ci = Pcom[ii] - P[ii]
        Pi = P[ii + 1] - Pcom[ii]

        Fx = Fcom[ii][0, :] + F[ii + 1][0, :] - M[ii] * gravity[0] - Fext[ii][0, :]
        Fy = Fcom[ii][1, :] + F[ii + 1][1, :] - M[ii] * gravity[1] - Fext[ii][1, :]
        Fz = (Fcom[ii][2, :] + F[ii + 1][2, :] - Fext[ii][2, :] +
              Ci[0, :] * F[ii][1, :] - Ci[1, :] * F[ii][0, :] +
              Pi[0, :] * F[ii + 1][1, :] - Pi[1, :] * F[ii + 1][0, :])

        F[ii] = ca.vertcat(Fx, Fy, Fz)

    return F

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