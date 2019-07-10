
# Imports
import numpy as np
from construct_graph import construct_complete_graph
from solve_MCFP import *
from draw import draw_sol


if __name__ == '__main__':
    # Initialize number of nodes and construct complete graph
    n = 10
    G, s, t = construct_complete_graph(n)

    # Draw nominal costs and deviations
    num_edges = np.shape(G.edges)[0]
    mu = np.random.uniform(0, 10, num_edges)
    delta = np.array([np.random.uniform(0, mu_ij) for mu_ij in mu])
    # Set Gamma - uncertainty ball size
    Gamma = 1

    # Solve MCFP (nominal model with costs = mu)
    P_nominal, x_nominal = solve_MCFP(G, s, t, mu)
    # Solve robust MCFP with different uncertainty sets
    P_inf, x_inf = solve_RMCFP_L_inf(G, s, t, mu, delta, Gamma)
    P_1, x_1 = solve_RMCFP_L_1(G, s, t, mu, delta, Gamma)
    P_2, x_2 = solve_RMCFP_L_2(G, s, t, mu, delta, Gamma)

    # --- Plot solution ---
    # Manually set a layout of graph
    pos = {
        0:  (0.15, 0.65), 1:  (0.27, 0.77), 2:  (0.39, 0.82), 3:  (0.52, 0.80),
        4:  (0.63, 0.74), 5:  (0.68, 0.61), 6:  (0.62, 0.51), 7:  (0.51, 0.46),
        8:  (0.38, 0.45), 9:  (0.25, 0.52)
    }

    draw_sol(G, pos, P_nominal)
    draw_sol(G, pos, P_inf)
    draw_sol(G, pos, P_1)
    draw_sol(G, pos, P_2)

