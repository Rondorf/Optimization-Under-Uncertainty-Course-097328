
import numpy as np
import picos as pic


def inventory_aarc(c, h, b, d_0, r):
    T = len(c)
    c_pic = pic.new_param('c', c)
    h_pic = pic.new_param('h', h)
    b_pic = pic.new_param('b', b)
    r_pic = pic.new_param('r', r)
    d0_pic = pic.new_param('d_0', d_0)

    # Define auxiliary parameters
    P_arr = []
    for t in range(T):
        P_t = np.concatenate((np.eye(t+1), np.zeros((t+1, T - (t + 1)))), axis=1)
        P_arr.append(pic.new_param('P' + str(t+1), P_t))

    ones_arr = []
    for t in range(T):
        ones_t = np.ones((t+1, 1))
        ones_arr.append(pic.new_param('1_' + str(t+1), ones_t))

    # Initialize picos problem.
    pb = pic.Problem()
    # Add variables.
    x0 = pb.add_variable("x0", T)
    x_arr = []
    for t in range(T-1):
        x_arr.append(pb.add_variable("x" + str(t + 1), t+1))
    v0 = pb.add_variable("v0", T)
    v_arr = []
    for t in range(T):
        v_arr.append(pb.add_variable("v" + str(t + 1), T))
    u = pb.add_variable("u")
    lamda = pb.add_variable("lambda")
    eta_arr = []
    for t in range(T):
        eta_arr.append(pb.add_variable("eta" + str(t + 1)))
    rho_arr = []
    for t in range(T):
        rho_arr.append(pb.add_variable("rho" + str(t + 1)))

    # Define auxiliary variables
    c_tilde = sum([c_pic[t + 1] * P_arr[t].T * x_arr[t] + v_arr[t] for t in range(T-1)]) + v_arr[-1]
    c_h_arr = []
    for t in range(T):
        if t == 0:
            c_h_arr.append(-h_pic[0] * P_arr[0].T * ones_arr[0] - v_arr[0])
        else:
            c_h_arr.append(h_pic[t] * sum([P_arr[s].T * x_arr[s] for s in range(t)]) -
                           h_pic[t] * P_arr[t].T * ones_arr[t] - v_arr[t])
    c_b_arr = []
    for t in range(T):
        if t == 0:
            c_b_arr.append(b_pic[0] * P_arr[0].T * ones_arr[0] - v_arr[0])
        else:
            c_b_arr.append(-b_pic[t] * sum([P_arr[s].T * x_arr[s] for s in range(t)]) +
                           b_pic[t] * P_arr[t].T * ones_arr[t] - v_arr[t])

    # Add constraints.
    pb.add_constraint(u >= (c_pic | x0) + (1 | v0) + (c_tilde | d0_pic) + lamda * r_pic)
    for t in range(T):
        pb.add_constraint(v0[t] >= h_pic[t] * ones_arr[t].T * P_arr[t] * x0 + (c_h_arr[t] | d0_pic) + eta_arr[t] * r_pic)
    for t in range(T):
        pb.add_constraint(v0[t] >= -b_pic[t] * ones_arr[t].T * P_arr[t] * x0 + (c_b_arr[t] | d0_pic) + rho_arr[t] * r_pic)
    pb.add_constraint(lamda >= pic.norm(c_tilde, 1))
    for t in range(T):
        pb.add_constraint(eta_arr[t] >= pic.norm(c_h_arr[t], 1))
    for t in range(T):
        pb.add_constraint(rho_arr[t] >= pic.norm(c_b_arr[t], 1))
    # Set objective.
    pb.set_objective("min", u)

    solution = pb.solve(solver='cvxopt')

    return pb, solution
