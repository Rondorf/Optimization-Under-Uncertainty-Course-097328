
import numpy as np
import picos as pic
from auxiliary import define_art_params, unpack_params_picos, convert_art_params_picos

MAX_MAN_HOUR = 3000


def solve_moment_based(params, r, mu, Sigma):
    # Unpack problem parameters
    c, q, xi_0, b, B = unpack_params_picos(params)
    r = pic.new_param('r', r)
    # Unpack artificial parameters
    E, c_tilde, E_tilde_arr, E_arr, e5 = define_art_params(c.get_value(), q.get_value(), xi_0)
    E, c_tilde, E_tilde_arr, E_arr, e5 = convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5)
    # Get dimensions.
    x_dim = len(c)
    xi_dim = len(xi_0)
    num_piecewise_lin_f = len(E_tilde_arr)
    # Initialize picos problem.
    pb = pic.Problem()

    # Add variables.
    x_tilde = pb.add_variable("x_tilde", x_dim + 1)
    t = pb.add_variable("t", 1)
    theta_arr = [pb.add_variable("theta_" + str(i + 1), 1) for i in range(num_piecewise_lin_f)]
    beta = pb.add_variable("beta", xi_dim)
    phi_arr = [pb.add_variable("phi_" + str(i + 1), xi_dim)
               for i in range(num_piecewise_lin_f)]
    Lambda = pb.add_variable("Lambda", (xi_dim, xi_dim), vtype='symmetric')
    lambda1_arr = [pb.add_variable("lambda1_" + str(i + 1), xi_dim)
                   for i in range(num_piecewise_lin_f)]
    lambda2_arr = [pb.add_variable("lambda2_" + str(i + 1), xi_dim)
                   for i in range(num_piecewise_lin_f)]
    eta1_arr = [pb.add_variable("eta1_" + str(i + 1), xi_dim)
                for i in range(2)]
    eta2_arr = [pb.add_variable("eta2_" + str(i + 1), xi_dim)
                for i in range(2)]

    # Add constraints.
    for i in range(num_piecewise_lin_f):
        pb.add_constraint((lambda1_arr[i] | (xi_0 + b)) - (lambda2_arr[i] | (xi_0 - b))
                          - (2 * phi_arr[i] | mu) + theta_arr[i] <= t)
        pb.add_constraint(lambda1_arr[i] - lambda2_arr[i] - 2 * phi_arr[i] + beta
                          == E_tilde_arr[i].T * x_tilde)
        pb.add_constraint((Lambda & phi_arr[i]) // (phi_arr[i].T & theta_arr[i]) >> 0)
        # positivity of lagrange multipliers
        pb.add_constraint(lambda1_arr[i] >= 0)
        pb.add_constraint(lambda2_arr[i] >= 0)

    for j in range(2):
        pb.add_constraint(abs(B.T * E_arr[j].T * x_tilde - B.T * (eta1_arr[j] - eta2_arr[j]))
                          <= MAX_MAN_HOUR - (x_tilde.T * E_arr[j] * xi_0 / r)
                          - ((eta1_arr[j] + eta2_arr[j]) | b / r))
        # positivity of lagrange multipliers
        pb.add_constraint(eta1_arr[j] >= 0)
        pb.add_constraint(eta2_arr[j] >= 0)

    pb.add_constraint(E.T * x_tilde >= 0)
    pb.add_constraint(e5 | x_tilde == -1)

    # Set objective.
    pb.set_objective("min", t + (beta | mu) + (Lambda | Sigma) - (c_tilde | x_tilde))

    solution = pb.solve(solver='cvxopt')
    opt_val = -pb.obj_value()
    opt_sol = np.array(solution['cvxopt_sol']['x'][:x_dim]).reshape(-1).round(5)
    return opt_sol, opt_val


def solve_moment_based_partial_lifting(params, r, mu, t, g):
    g = pic.new_param('g', g)
    # Unpack problem parameters
    c, q, xi_0, b, B = unpack_params_picos(params)
    r = pic.new_param('r', r)
    # Unpack artificial parameters
    E, c_tilde, E_tilde_arr, E_arr, e5 = define_art_params(c.get_value(), q.get_value(), xi_0)
    E, c_tilde, E_tilde_arr, E_arr, e5 = convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5)
    # Get dimensions.
    x_dim = len(c)
    xi_dim = len(xi_0)
    num_piecewise_lin_f = len(E_tilde_arr)
    t_dim = len(t)
    # Initialize picos problem.
    pb = pic.Problem()

    # Add variables.
    x_tilde = pb.add_variable("x_tilde", x_dim + 1)
    kappa = pb.add_variable("kappa", 1)
    beta = pb.add_variable("beta", xi_dim)
    delta = pb.add_variable("delta", t_dim)
    alpha_arr = []
    for i in range(t_dim):
        for j in range(num_piecewise_lin_f):
            alpha_arr.append(pb.add_variable("alpha_" + str(i + 1) + "_" + str(j + 1), 3))
    lambda1_arr = [pb.add_variable("lambda1_" + str(i + 1), xi_dim)
                   for i in range(num_piecewise_lin_f)]
    lambda2_arr = [pb.add_variable("lambda2_" + str(i + 1), xi_dim)
                   for i in range(num_piecewise_lin_f)]
    eta1_arr = [pb.add_variable("eta1_" + str(i + 1), xi_dim)
                for i in range(2)]
    eta2_arr = [pb.add_variable("eta2_" + str(i + 1), 10)
                for i in range(2)]

    # Add constraints.
    for j in range(num_piecewise_lin_f):
        pb.add_constraint((lambda1_arr[j] | (xi_0 + b)) - (lambda2_arr[j] | (xi_0 - b)) +
                          0.5 * (sum(alpha_arr[j::num_piecewise_lin_f])[2] -
                                 sum(alpha_arr[j::num_piecewise_lin_f])[1]) <= kappa)
        pb.add_constraint(0.5 * (sum(alpha_arr[j::num_piecewise_lin_f])[2] +
                                 sum(alpha_arr[j::num_piecewise_lin_f])[1]) == delta)
        pb.add_constraint(lambda1_arr[j] - lambda2_arr[j] -
                          sum([alpha[0] * g[i, :].T for alpha, i
                               in zip(alpha_arr[j::num_piecewise_lin_f], range(t_dim))]) + beta
                          == E_tilde_arr[j].T * x_tilde)
        # positivity of lagrange multipliers
        pb.add_constraint(lambda1_arr[j] >= 0)
        pb.add_constraint(lambda2_arr[j] >= 0)
        # conic dual variables (alpha in L^3)
        for alpha_ij in alpha_arr[j::num_piecewise_lin_f]:
            pb.add_constraint(abs(alpha_ij[0] // alpha_ij[1]) <= alpha_ij[2])

    for k in range(2):
        pb.add_constraint(abs(B.T * E_arr[k].T * x_tilde - B.T * (eta1_arr[k] - eta2_arr[k]))
                          <= MAX_MAN_HOUR - (x_tilde.T * E_arr[k] * xi_0 / r)
                          - ((eta1_arr[k] + eta2_arr[k]) | b / r))
        # positivity of lagrange multipliers
        pb.add_constraint(eta1_arr[k] >= 0)
        pb.add_constraint(eta2_arr[k] >= 0)

    pb.add_constraint(E.T * x_tilde >= 0)
    pb.add_constraint(e5 | x_tilde == -1)

    # Set objective.
    pb.set_objective("min", kappa + (beta | mu) + (delta | t) - (c_tilde | x_tilde))

    solution = pb.solve(solver='cvxopt')
    opt_val = -pb.obj_value()
    opt_sol = np.array(solution['cvxopt_sol']['x'][:x_dim]).reshape(-1).round(5)
    return opt_sol, opt_val


def solve_data_driven_moment_based(params, r, gamma1, gamma2, mu, Sigma):
    # Unpack problem parameters
    c, q, xi_0, b, B = unpack_params_picos(params)
    r = pic.new_param('r', r)
    # Unpack artificial parameters
    E, c_tilde, E_tilde_arr, E_arr, e5 = define_art_params(c.get_value(), q.get_value(), xi_0)
    E, c_tilde, E_tilde_arr, E_arr, e5 = convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5)
    # Get dimensions.
    x_dim = len(c)
    xi_dim = len(xi_0)
    num_piecewise_lin_f = len(E_tilde_arr)
    # Initialize picos problem.
    pb = pic.Problem()

    # Add variables.
    x_tilde = pb.add_variable("x_tilde", x_dim + 1)
    t = pb.add_variable("t", 1)
    beta = pb.add_variable("beta", 1)
    theta1_arr = [pb.add_variable("theta1_" + str(i + 1), 1) for i in range(num_piecewise_lin_f)]
    phi1_arr = [pb.add_variable("phi1_" + str(i + 1), xi_dim)
                for i in range(num_piecewise_lin_f)]
    phi_sigma_arr = [pb.add_variable("phi_sigma_" + str(i + 1), xi_dim)
                     for i in range(num_piecewise_lin_f)]
    Lambda = pb.add_variable("Lambda", (xi_dim, xi_dim), vtype='symmetric')
    Gamma_sigma_tilde_arr = [pb.add_variable("Gamma_sigma_tilde_" + str(i + 1),
                                             (xi_dim, xi_dim), vtype='symmetric')
                             for i in range(num_piecewise_lin_f)]
    lambda1_arr = [pb.add_variable("lambda1_" + str(i + 1), xi_dim)
                   for i in range(num_piecewise_lin_f)]
    lambda2_arr = [pb.add_variable("lambda2_" + str(i + 1), xi_dim)
                   for i in range(num_piecewise_lin_f)]
    eta1_arr = [pb.add_variable("eta1_" + str(i + 1), xi_dim)
                for i in range(2)]
    eta2_arr = [pb.add_variable("eta2_" + str(i + 1), 10)
                for i in range(2)]

    # Add constraints.
    for i in range(num_piecewise_lin_f):
        pb.add_constraint((lambda1_arr[i] | (xi_0 + b)) - (lambda2_arr[i] | (xi_0 - b))
                          - (2 * (phi1_arr[i] + phi_sigma_arr[i]) | mu) +
                          (Gamma_sigma_tilde_arr[i] | Sigma) + theta1_arr[i] <= t)
        pb.add_constraint(lambda1_arr[i] - lambda2_arr[i] - 2 * (phi1_arr[i] + phi_sigma_arr[i])
                          == E_tilde_arr[i].T * x_tilde)
        pb.add_constraint((Lambda & phi1_arr[i]) // (phi1_arr[i].T & theta1_arr[i]) >> 0)
        pb.add_constraint((Gamma_sigma_tilde_arr[i] & phi_sigma_arr[i]) //
                          (phi_sigma_arr[i].T & beta) >> 0)
        # positivity of lagrange multipliers
        pb.add_constraint(lambda1_arr[i] >= 0)
        pb.add_constraint(lambda2_arr[i] >= 0)

    for j in range(2):
        pb.add_constraint(abs(B.T * E_arr[j].T * x_tilde - B.T * (eta1_arr[j] - eta2_arr[j]))
                          <= MAX_MAN_HOUR - (x_tilde.T * E_arr[j] * xi_0 / r)
                          - ((eta1_arr[j] + eta2_arr[j]) | b / r))
        # positivity of lagrange multipliers
        pb.add_constraint(eta1_arr[j] >= 0)
        pb.add_constraint(eta2_arr[j] >= 0)

    pb.add_constraint(E.T * x_tilde >= 0)
    pb.add_constraint(e5 | x_tilde == -1)

    # Set objective.
    pb.set_objective("min", t + (beta * gamma2) + gamma1 * (Lambda | Sigma) - (c_tilde | x_tilde))

    solution = pb.solve(solver='cvxopt')
    opt_val = -pb.obj_value()
    opt_sol = np.array(solution['cvxopt_sol']['x'][:x_dim]).reshape(-1).round(5)
    return opt_sol, opt_val


def solve_1_wasserstein(params, Xi, r, eps_N):
    '''
        Xi is essential as input, since we need the values of the realizations rather
        than some statistics of (e.g. mean, covariance, etc.)
    '''
    Xi_pic = pic.new_param('Xi', Xi)
    N = Xi.shape[0]
    # Unpack problem parameters
    c, q, xi_0, b, B = unpack_params_picos(params)
    r = pic.new_param('r', r)
    # Unpack artificial parameters
    E, c_tilde, E_tilde_arr, E_arr, e5 = define_art_params(c.get_value(), q.get_value(), xi_0)
    E, c_tilde, E_tilde_arr, E_arr, e5 = convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5)
    # Get dimensions.
    x_dim = len(c)
    xi_dim = len(xi_0)
    num_piecewise_lin_f = len(E_tilde_arr)
    # Initialize picos problem.
    pb = pic.Problem()

    # Add variables.
    x_tilde = pb.add_variable("x_tilde", x_dim + 1)
    t = [pb.add_variable("t_" + str(i + 1), 1) for i in range(N)]
    lamda = pb.add_variable("lamda", 1)

    lambda1_arr = []
    lambda2_arr = []
    for i in range(N):
        for j in range(num_piecewise_lin_f):
            lambda1_arr.append(pb.add_variable("lambda1_" + str(i + 1) + "_" + str(j + 1), xi_dim))
            lambda2_arr.append(pb.add_variable("lambda2_" + str(i + 1) + "_" + str(j + 1), xi_dim))
    eta1_arr = [pb.add_variable("eta1_" + str(i + 1), xi_dim)
                for i in range(2)]
    eta2_arr = [pb.add_variable("eta2_" + str(i + 1), 10)
                for i in range(2)]

    # Add constraints.
    for i in range(N):
        for j in range(num_piecewise_lin_f):
            Xi_i_hat = Xi_pic[i, :].T
            pb.add_constraint(((E_tilde_arr[j] * Xi_i_hat) | x_tilde) -
                              ((Xi_i_hat - xi_0 - b) | lambda1_arr[(num_piecewise_lin_f * i) + j]) +
                              ((Xi_i_hat - xi_0 + b) | lambda2_arr[(num_piecewise_lin_f * i) + j])
                              <= t[i])
            pb.add_constraint(abs((E_tilde_arr[j].T * x_tilde)
                              - lambda1_arr[(num_piecewise_lin_f * i) + j]
                              + lambda2_arr[(num_piecewise_lin_f * i) + j]) <= lamda)
            # positivity of lagrange multipliers
            pb.add_constraint(lambda1_arr[(num_piecewise_lin_f * i) + j] >= 0)
            pb.add_constraint(lambda2_arr[(num_piecewise_lin_f * i) + j] >= 0)

    for k in range(2):
        pb.add_constraint(abs(B.T * E_arr[k].T * x_tilde - B.T * (eta1_arr[k] - eta2_arr[k]))
                          <= MAX_MAN_HOUR - (x_tilde.T * E_arr[k] * xi_0 / r)
                          - ((eta1_arr[k] + eta2_arr[k]) | b / r))
        # positivity of lagrange multipliers
        pb.add_constraint(eta1_arr[k] >= 0)
        pb.add_constraint(eta2_arr[k] >= 0)

    pb.add_constraint(lamda >= 0)
    pb.add_constraint(E.T * x_tilde >= 0)
    pb.add_constraint(e5 | x_tilde == -1)

    # Set objective.
    pb.set_objective("min", (1 / N) * sum(t[i] for i in range(N))
                     + lamda * eps_N - (c_tilde | x_tilde))

    solution = pb.solve(solver='cvxopt')
    opt_val = -pb.obj_value()
    opt_sol = np.array(solution['cvxopt_sol']['x'][:x_dim]).reshape(-1).round(5)
    return opt_sol, opt_val


def solve_SRO(params, Xi, eps_N):
    '''
        Xi is essential as input, since we need the values of the realizations rather
        than some statistics of (e.g. mean, covariance, etc.)
    '''
    Xi_pic = pic.new_param('Xi', Xi)
    eps_N = pic.new_param('eps_N', eps_N)
    N = Xi.shape[0]
    # Unpack problem parameters
    c, q, xi_0, b, B = unpack_params_picos(params)
    # Unpack artificial parameters
    E, c_tilde, E_tilde_arr, E_arr, e5 = define_art_params(c.get_value(), q.get_value(), xi_0)
    E, c_tilde, E_tilde_arr, E_arr, e5 = convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5)
    # Get dimensions.
    x_dim = len(c)
    xi_dim = len(xi_0)
    num_piecewise_lin_f = len(E_tilde_arr)
    # Initialize picos problem.
    pb = pic.Problem()

    # Add variables.
    x_tilde = pb.add_variable("x_tilde", x_dim + 1)
    t = pb.add_variable("t", N)

    lambda_arr = []
    lambda1_arr = []
    lambda2_arr = []
    eta_arr = []
    eta1_arr = []
    eta2_arr = []
    for i in range(N):
        for j in range(num_piecewise_lin_f):
            lambda_arr.append(pb.add_variable("lambda_" + str(i + 1) + "_" + str(j + 1), 1))
            lambda1_arr.append(pb.add_variable("lambda1_" + str(i + 1) + "_" + str(j + 1), xi_dim))
            lambda2_arr.append(pb.add_variable("lambda2_" + str(i + 1) + "_" + str(j + 1), xi_dim))
        for k in range(2):
            eta_arr.append(pb.add_variable("eta_" + str(i + 1) + "_" + str(k + 1), 1))
            eta1_arr.append(pb.add_variable("eta1_" + str(i + 1) + "_" + str(k + 1), xi_dim))
            eta2_arr.append(pb.add_variable("eta2_" + str(i + 1) + "_" + str(k + 1), xi_dim))

    # Add constraints.
    for i in range(N):
        Xi_i_hat = Xi_pic[i, :].T
        for j in range(num_piecewise_lin_f):
            pb.add_constraint(((E_tilde_arr[j] * Xi_i_hat) | x_tilde) -
                              ((Xi_i_hat - xi_0 - b) | lambda1_arr[(num_piecewise_lin_f * i) + j]) +
                              ((Xi_i_hat - xi_0 + b) | lambda2_arr[(num_piecewise_lin_f * i) + j]) +
                              (lambda_arr[(num_piecewise_lin_f * i) + j] * eps_N)
                              <= t[i])
            pb.add_constraint(abs((E_tilde_arr[j].T * x_tilde)
                                  - lambda1_arr[(num_piecewise_lin_f * i) + j]
                                  + lambda2_arr[(num_piecewise_lin_f * i) + j])
                              <= lambda_arr[(num_piecewise_lin_f * i) + j])
            # positivity of lagrange multipliers
            pb.add_constraint(lambda_arr[(num_piecewise_lin_f * i) + j] >= 0)
            pb.add_constraint(lambda1_arr[(num_piecewise_lin_f * i) + j] >= 0)
            pb.add_constraint(lambda2_arr[(num_piecewise_lin_f * i) + j] >= 0)

        for k in range(2):
            pb.add_constraint(((E_arr[k] * Xi_i_hat) | x_tilde) -
                              ((Xi_i_hat - xi_0 - b) | eta1_arr[(2 * i) + k]) +
                              ((Xi_i_hat - xi_0 + b) | eta2_arr[(2 * i) + k]) +
                              (eta_arr[(2 * i) + k] * eps_N)
                              <= MAX_MAN_HOUR)
            pb.add_constraint(abs((E_arr[k].T * x_tilde)
                                  - eta1_arr[(2 * i) + k]
                                  + eta2_arr[(2 * i) + k])
                              <= eta_arr[(2 * i) + k])
            # positivity of lagrange multipliers
            pb.add_constraint(eta_arr[(2 * i) + k] >= 0)
            pb.add_constraint(eta1_arr[(2 * i) + k] >= 0)
            pb.add_constraint(eta2_arr[(2 * i) + k] >= 0)

    pb.add_constraint(E.T * x_tilde >= 0)
    pb.add_constraint(e5 | x_tilde == -1)

    # Set objective.
    pb.set_objective("min", (1 / N) * (1 | t) - (c_tilde | x_tilde))

    solution = pb.solve(solver='cvxopt')
    opt_val = -pb.obj_value()
    opt_sol = np.array(solution['cvxopt_sol']['x'][:x_dim]).reshape(-1).round(5)
    return opt_sol, opt_val


def solve_SAA(params, Xi, r):
    N = Xi.shape[0]
    Xi_pic = [pic.new_param('xi_' + str(i + 1), Xi[i]) for i in range(N)]
    # Unpack problem parameters
    c, q, xi_0, b, B = unpack_params_picos(params)
    r = pic.new_param('r', r)
    # Unpack artificial parameters
    E, c_tilde, E_tilde_arr, E_arr, e5 = define_art_params(c.get_value(), q.get_value(), xi_0)
    E, c_tilde, E_tilde_arr, E_arr, e5 = convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5)
    # Get dimensions.
    x_dim = len(c)
    xi_dim = len(xi_0)
    num_piecewise_lin_f = len(E_tilde_arr)
    # Initialize picos problem.
    pb = pic.Problem()

    # Add variables.
    x_tilde = pb.add_variable("x_tilde", x_dim + 1)
    t = pb.add_variable("t", N)
    eta1_arr = [pb.add_variable("eta1_" + str(i + 1), xi_dim)
                for i in range(2)]
    eta2_arr = [pb.add_variable("eta2_" + str(i + 1), 10)
                for i in range(2)]

    # Add constraints.
    for i in range(N):
        for j in range(num_piecewise_lin_f):
            pb.add_constraint((E_tilde_arr[j] * Xi_pic[i]) | x_tilde <= t[i])

    for j in range(2):
        pb.add_constraint(abs(B.T * E_arr[j].T * x_tilde - B.T * (eta1_arr[j] - eta2_arr[j]))
                          <= MAX_MAN_HOUR - (x_tilde.T * E_arr[j] * xi_0 / r)
                          - ((eta1_arr[j] + eta2_arr[j]) | b / r))
        # positivity of lagrange multipliers
        pb.add_constraint(eta1_arr[j] >= 0)
        pb.add_constraint(eta2_arr[j] >= 0)

    pb.add_constraint(E.T * x_tilde >= 0)
    pb.add_constraint(e5 | x_tilde == -1)

    # Set objective.
    pb.set_objective("min", (1 / N) * (1 | t) - (c_tilde | x_tilde))

    solution = pb.solve(solver='cvxopt')
    opt_val = -pb.obj_value()
    opt_sol = np.array(solution['cvxopt_sol']['x'][:x_dim]).reshape(-1).round(5)    # round 5 digits
    return opt_sol, opt_val

