
import numpy as np
from estimate_from_data import *
from solve_DRO import *
from auxiliary import *


if __name__ == '__main__':
    # Parameters
    c = np.array([12., 20., 18., 40.])
    q = np.array([5., 10.])
    xi_0 = np.array([4., 9., 7., 10., 1., 1., 3., 40., 6000., 4000.])
    b = np.array([.5, 1., 1., 1., .2, .2, .5, 4., 300., 150.])
    B = np.diag(b)
    # Pack parameter
    params = [c, q, xi_0, b]

    # draw fixed test set
    N_test = 10000
    Xi_test = draw_xi(N_test)

    # draw training set
    N = 100          # N = [10, 100, 1000]
    Xi_train = draw_xi(N)

    # compute statistics (moments)
    mu, Sigma = empirical_mean(Xi_train), empirical_cov(Xi_train)
    # g = get_binary_vectors(len(xi_0))   # binary vectors for computing t
    # t = empirical_t(Xi_train, g)

    # constraints and re-formulation parameters
    r = 1           # r = [1, 2, 5]
    gamma1 = 1      # gamma1 = [0.5, 1, 2]
    gamma2 = 1     # gamma2 = [0.5, 1, 2]

    # Moment-based ambiguity set - 1
    sol_moment_based, val_moment_based = solve_moment_based(params, r, mu, Sigma)
    # Moment-based ambiguity set with partial lifting - 2
    # sol_moment_based_w_part_lift, val_moment_based_w_part_lift = solve_moment_based_partial_lifting(params, r, mu, t, g)
    # Data-driven moment-based ambiguity set - 3
    sol_dd_moment_based, val_dd_moment_based = solve_data_driven_moment_based(params, r, gamma1, gamma2, mu, Sigma)

    # compare feasibility
    feas_mb_perc, feas_mb = is_feasible(sol_moment_based, Xi_test, c, q, xi_0)
    # feas_mbpl_perc, feas_mbpl = is_feasible(sol_moment_based_w_part_lift, Xi_test, c, q, xi_0)
    feas_ddmb_perc, feas_ddmb = is_feasible(sol_dd_moment_based, Xi_test, c, q, xi_0)
    # compare average profit
    profit_mb = average_profit(sol_moment_based, Xi_test[feas_mb], c, q, xi_0)
    # profit_mbpl = average_profit(sol_moment_based_w_part_lift, Xi_test[feas_mbpl], c, q, xi_0)
    profit_ddmb = average_profit(sol_dd_moment_based, Xi_test[feas_ddmb], c, q, xi_0)

    # draw training set
    N = 20         # N = [10, 20, 50]
    Xi_train = draw_xi(N)
    # constraints and re-formulation parameters
    eps_N = 1.       # eps_N = [1, 2, 5]
    # 1-wasserstein ambiguity set - 4
    sol_1_wasserstein, val_1_wasserstein = solve_1_wasserstein(params, Xi_train, r, eps_N)
    # SRO - 5
    sol_sro, val_sro = solve_SRO(params, Xi_train, eps_N)
    # SAA
    sol_saa, val_saa = solve_SAA(params, Xi_train, r)

    # compare feasibility
    feas_1_wasserstein_perc, feas_1_wasserstein = is_feasible(sol_1_wasserstein, Xi_test, c, q, xi_0)
    feas_sro_perc, feas_sro = is_feasible(sol_sro, Xi_test, c, q, xi_0)
    feas_saa_perc, feas_saa = is_feasible(sol_saa, Xi_test, c, q, xi_0)
    # compare average profit
    profit_1_wasserstein = average_profit(sol_1_wasserstein, Xi_test[feas_1_wasserstein], c, q, xi_0)
    profit_sro = average_profit(sol_sro, Xi_test[feas_sro], c, q, xi_0)
    profit_saa = average_profit(sol_saa, Xi_test[feas_saa], c, q, xi_0)

