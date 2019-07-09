
# Imports
from solve_MCFP import *
from construct_graph import construct_complete_graph


def extract_statistics(array):
    '''
    This function extract statistics of an array - mean, std, median, max
    and probability of being greater than or equal to some threshold
    :param array: numpy array
    :return:
    '''
    mean_val = np.mean(array)
    std_val = np.std(array)
    median_val = np.median(array)
    max_val = np.max(array)
    return mean_val, std_val, median_val, max_val


def compute_prob_of_higher_threshold(array, threshold):
    '''
    This function computes the probability of being greater
    than or equal to some threshold (empirical estimate)
    :param array: numpy array
    :return:
    '''

    p_higher_than_threshold = np.mean(array > threshold)
    return p_higher_than_threshold


# Initialize number of nodes and construct complete graph
n = 5
G, s, t = construct_complete_graph(n)

# Draw nominal costs and deviations
num_edges = np.shape(G.edges)[0]
mu = np.random.uniform(0, 10, num_edges)
delta = np.array([np.random.uniform(0, mu_ij) for mu_ij in mu])

# Solve robust MCFP
P_L_inf, x_L_inf = solve_RMCFP_L_inf(G, s, t, mu, delta, Gamma=1)
P_L_1, x_L_1 = solve_RMCFP_L_1(G, s, t, mu, delta, Gamma=1)
P_L_2, x_L_2 = solve_RMCFP_L_2(G, s, t, mu, delta, Gamma=1)
# Solve non-robust MCFP
P_nominal, x_nominal = solve_MCFP(G, s, t, mu)

# Parameters for beta distribution
beta_params = [(0.5, 0.5), (5, 1), (2, 2), (2, 5)]

num_experiments = len(beta_params)
num_simulations = 10000
# Initialize dictionary of objective values for each of the 4 solutions (3 uncertainty sets and nominal).
# Each dictionary contains 10000 values for each of the beta parameters pairs (10000 x 4 in total)
objective_values_L_inf = {}
objective_values_L_1 = {}
objective_values_L_2 = {}
objective_values_nominal = {}

for params in beta_params:
    # Initialize list for every beta parameters pair
    objective_values_L_inf[params] = []
    objective_values_L_1[params] = []
    objective_values_L_2[params] = []
    objective_values_nominal[params] = []
    for simulation in range(num_simulations):
        # Generate costs using beta distribution
        costs = np.array([2 * np.random.beta(params[0], params[1]) * delta_ij + (mu_ij - delta_ij)
                          for (delta_ij, mu_ij) in zip(delta, mu)])
        # Compute objective value for each of the optimal solutions (different uncertainty sets and nominal)
        objective_values_L_inf[params].append(np.dot(costs, x_L_inf))
        objective_values_L_1[params].append(np.dot(costs, x_L_1))
        objective_values_L_2[params].append(np.dot(costs, x_L_2))
        objective_values_nominal[params].append(np.dot(costs, x_nominal))
    # Convert to numpy array
    objective_values_L_inf[params] = np.array(objective_values_L_inf[params])
    objective_values_L_1[params] = np.array(objective_values_L_1[params])
    objective_values_L_2[params] = np.array(objective_values_L_2[params])
    objective_values_nominal[params] = np.array(objective_values_nominal[params])


# unpack simulation values of four experiments, for every problem solution
objective_exp1_L_inf = objective_values_L_inf[beta_params[0]]
objective_exp2_L_inf = objective_values_L_inf[beta_params[1]]
objective_exp3_L_inf = objective_values_L_inf[beta_params[2]]
objective_exp4_L_inf = objective_values_L_inf[beta_params[3]]

objective_exp1_L_1 = objective_values_L_1[beta_params[0]]
objective_exp2_L_1 = objective_values_L_1[beta_params[1]]
objective_exp3_L_1 = objective_values_L_1[beta_params[2]]
objective_exp4_L_1 = objective_values_L_1[beta_params[3]]

objective_exp1_L_2 = objective_values_L_2[beta_params[0]]
objective_exp2_L_2 = objective_values_L_2[beta_params[1]]
objective_exp3_L_2 = objective_values_L_2[beta_params[2]]
objective_exp4_L_2 = objective_values_L_2[beta_params[3]]

objective_exp1_nominal = objective_values_nominal[beta_params[0]]
objective_exp2_nominal = objective_values_nominal[beta_params[1]]
objective_exp3_nominal = objective_values_nominal[beta_params[2]]
objective_exp4_nominal = objective_values_nominal[beta_params[3]]


# optimal values of robust solutions and nominal solution
optimal_inf = P_L_inf.obj_value()
optimal_r1 = P_L_1.obj_value()
optimal_r2 = P_L_2.obj_value()
optimal_nominal = P_nominal.obj_value()

# extract statistics of four experiments, for every problem solution
mean_val_exp1_L_inf, std_val_exp1_L_inf, median_val_exp1_L_inf, max_val_exp1_L_inf = \
    extract_statistics(objective_exp1_L_inf)
mean_val_exp2_L_inf, std_val_exp2_L_inf, median_val_exp2_L_inf, max_val_exp2_L_inf = \
    extract_statistics(objective_exp2_L_inf)
mean_val_exp3_L_inf, std_val_exp3_L_inf, median_val_exp3_L_inf, max_val_exp3_L_inf = \
    extract_statistics(objective_exp3_L_inf)
mean_val_exp4_L_inf, std_val_exp4_L_inf, median_val_exp4_L_inf, max_val_exp4_L_inf = \
    extract_statistics(objective_exp4_L_inf)

mean_val_exp1_L_1, std_val_exp1_L_1, median_val_exp1_L_1, max_val_exp1_L_1 = \
    extract_statistics(objective_exp1_L_1)
mean_val_exp2_L_1, std_val_exp2_L_1, median_val_exp2_L_1, max_val_exp2_L_1 = \
    extract_statistics(objective_exp2_L_1)
mean_val_exp3_L_1, std_val_exp3_L_1, median_val_exp3_L_1, max_val_exp3_L_1 = \
    extract_statistics(objective_exp3_L_1)
mean_val_exp4_L_1, std_val_exp4_L_1, median_val_exp4_L_1, max_val_exp4_L_1 = \
    extract_statistics(objective_exp4_L_1)

mean_val_exp1_L_2, std_val_exp1_L_2, median_val_exp1_L_2, max_val_exp1_L_2 = \
    extract_statistics(objective_exp1_L_2)
mean_val_exp2_L_2, std_val_exp2_L_2, median_val_exp2_L_2, max_val_exp2_L_2 = \
    extract_statistics(objective_exp2_L_2)
mean_val_exp3_L_2, std_val_exp3_L_2, median_val_exp3_L_2, max_val_exp3_L_2 = \
    extract_statistics(objective_exp3_L_2)
mean_val_exp4_L_2, std_val_exp4_L_2, median_val_exp4_L_2, max_val_exp4_L_2 = \
    extract_statistics(objective_exp4_L_2)

mean_val_exp1_nominal, std_val_exp1_nominal, median_val_exp1_nominal, max_val_exp1_nominal = \
    extract_statistics(objective_exp1_nominal)
mean_val_exp2_nominal, std_val_exp2_nominal, median_val_exp2_nominal, max_val_exp2_nominal = \
    extract_statistics(objective_exp2_nominal)
mean_val_exp3_nominal, std_val_exp3_nominal, median_val_exp3_nominal, max_val_exp3_nominal = \
    extract_statistics(objective_exp3_nominal)
mean_val_exp4_nominal, std_val_exp4_nominal, median_val_exp4_nominal, max_val_exp4_nominal = \
    extract_statistics(objective_exp4_nominal)

# compute probability of being higher than expected value - L_inf robust
p_higher_than_exp1_L_inf = compute_prob_of_higher_threshold(objective_exp1_L_inf, optimal_inf)
p_higher_than_exp2_L_inf = compute_prob_of_higher_threshold(objective_exp2_L_inf, optimal_inf)
p_higher_than_exp3_L_inf = compute_prob_of_higher_threshold(objective_exp3_L_inf, optimal_inf)
p_higher_than_exp4_L_inf = compute_prob_of_higher_threshold(objective_exp4_L_inf, optimal_inf)

# compute probability of being higher than expected value - L_1 robust
p_higher_than_exp1_L_1 = compute_prob_of_higher_threshold(objective_exp1_L_1, optimal_r1)
p_higher_than_exp2_L_1 = compute_prob_of_higher_threshold(objective_exp2_L_1, optimal_r1)
p_higher_than_exp3_L_1 = compute_prob_of_higher_threshold(objective_exp3_L_1, optimal_r1)
p_higher_than_exp4_L_1 = compute_prob_of_higher_threshold(objective_exp4_L_1, optimal_r1)

# compute probability of being higher than expected value - L_2 robust
p_higher_than_exp1_L_2 = compute_prob_of_higher_threshold(objective_exp1_L_2, optimal_r2)
p_higher_than_exp2_L_2 = compute_prob_of_higher_threshold(objective_exp2_L_2, optimal_r2)
p_higher_than_exp3_L_2 = compute_prob_of_higher_threshold(objective_exp3_L_2, optimal_r2)
p_higher_than_exp4_L_2 = compute_prob_of_higher_threshold(objective_exp4_L_2, optimal_r2)

# compute probability of being higher than expected value - nominal solution (non-robust)
p_higher_than_exp1_nominal = compute_prob_of_higher_threshold(objective_exp1_nominal, optimal_nominal)
p_higher_than_exp2_nominal = compute_prob_of_higher_threshold(objective_exp2_nominal, optimal_nominal)
p_higher_than_exp3_nominal = compute_prob_of_higher_threshold(objective_exp3_nominal, optimal_nominal)
p_higher_than_exp4_nominal = compute_prob_of_higher_threshold(objective_exp4_nominal, optimal_nominal)

