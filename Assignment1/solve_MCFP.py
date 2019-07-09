
# Imports
import picos as pic
import numpy as np


def solve_MCFP(G, s, t, costs):
    '''
     This function solves the MCFP
    :param G: bidirectional graph object
    :param s: source node (0)
    :param t: sink (terminal) node (N-1)
    :param costs: edges costs
    :return:
    '''
    # Define problem using PICOS
    P = pic.Problem()

    # Add the flow variables
    x = {}
    for e in G.edges():
        x[e] = P.add_variable('x[{0}]'.format(e), 1)

    # --- Add constraints ---
    # Enforce flow conservation
    P.add_list_of_constraints(
        [pic.sum([x[i, j] for i in G.predecessors(j)], 'i', 'pred(j)')
         == pic.sum([x[j, k] for k in G.successors(j)], 'k', 'succ(j)')
         for j in G.nodes() if j not in (s, t)], 'i', 'nodes-(s,t)')

    # Set source flow at s
    P.add_constraint(
        pic.sum([x[s, j] for j in G.successors(s)], 'j', 'succ(s)') == 1)

    # Set sink flow at t
    P.add_constraint(
        pic.sum([x[i, t] for i in G.predecessors(t)], 'i', 'pred(t)') == 1)

    # Enforce edge capacities
    P.add_list_of_constraints(
        [x[e[:2]] <= e[2]['capacity'] for e in G.edges(data=True)],  # list of constraints
        [('e', 2)],  # e is a double index
        'edges')  # set the index belongs to

    # Enforce flow non-negativity
    P.add_list_of_constraints(
        [x[e] >= 0 for e in G.edges()],  # list of constraints
        [('e', 2)],  # e is a double index
        'edges')  # set the index belongs to

    # Define objective and solve
    objective = pic.sum([cost_ij * x_ij for (cost_ij, x_ij) in zip(costs, x.values())])
    P.set_objective('min', objective)
    sol = P.solve(verbose=0, solver='cvxopt')

    # Unpack solution vector
    x = np.array(sol['cvxopt_sol']['x']).reshape(-1)

    return P, x


def solve_RMCFP_L_inf(G, s, t, mu, delta, Gamma):
    '''
    This function solves the Robust MCFP with L_inf uncertainty set
    :param G: bidirectional graph object
    :param s: source node (0)
    :param t: sink (terminal) node (N-1)
    :param mu: nominal edges costs
    :param delta: costs deviations (uncertainty amplification)
    :param Gamma: uncertainty ball size
    :return:
    '''
    # Define problem using PICOS
    P = pic.Problem()

    # Add the flow variables
    x = {}
    for e in G.edges():
        x[e] = P.add_variable('x[{0}]'.format(e), 1)

    # --- Add constraints ---
    # Enforce flow conservation
    P.add_list_of_constraints(
        [pic.sum([x[i, j] for i in G.predecessors(j)], 'i', 'pred(j)')
         == pic.sum([x[j, k] for k in G.successors(j)], 'k', 'succ(j)')
         for j in G.nodes() if j not in (s, t)], 'i', 'nodes-(s,t)')

    # Set source flow at s
    P.add_constraint(
        pic.sum([x[s, j] for j in G.successors(s)], 'j', 'succ(s)') == 1)

    # Set sink flow at t
    P.add_constraint(
        pic.sum([x[i, t] for i in G.predecessors(t)], 'i', 'pred(t)') == 1)

    # Enforce edge capacities
    P.add_list_of_constraints(
        [x[e[:2]] <= e[2]['capacity'] for e in G.edges(data=True)],  # list of constraints
        [('e', 2)],  # e is a double index
        'edges')  # set the index belongs to

    # Enforce flow non-negativity
    P.add_list_of_constraints(
        [x[e] >= 0 for e in G.edges()],  # list of constraints
        [('e', 2)],  # e is a double index
        'edges')  # set the index belongs to

    # Define objective and solve
    costs = mu + Gamma * delta
    objective = pic.sum([cost_ij * x_ij for (cost_ij, x_ij) in zip(costs, x.values())])
    P.set_objective('min', objective)
    sol = P.solve(verbose=0, solver='cvxopt')

    # Unpack solution vector
    x = np.array(sol['cvxopt_sol']['x']).reshape(-1)

    return P, x


def solve_RMCFP_L_1(G, s, t, mu, delta, Gamma):
    '''
        This function solves the Robust MCFP with L_1 uncertainty set
        :param G: bidirectional graph object
        :param s: source node (0)
        :param t: sink (terminal) node (N-1)
        :param mu: nominal edges costs
        :param delta: costs deviations (uncertainty amplification)
        :param Gamma: uncertainty ball size
        :return:
        '''
    # Define problem using PICOS
    P = pic.Problem()

    # Add the flow variables and auxiliary variable
    x = {}
    for e in G.edges():
        x[e] = P.add_variable('x[{0}]'.format(e), 1)
    v = P.add_variable('v', 1)

    # --- Add constraints ---
    # Robust feasibility constraint (its equivalent)
    mu_dot_x = pic.sum([mu_ij * x_ij for (mu_ij, x_ij) in zip(mu, x.values())])
    P.add_list_of_constraints(
        [Gamma * delta_ij * x[e] + mu_dot_x <= v
         for (e, delta_ij) in zip(G.edges(), delta)])

    # Enforce flow conservation
    P.add_list_of_constraints(
        [pic.sum([x[i, j] for i in G.predecessors(j)], 'i', 'pred(j)')
         == pic.sum([x[j, k] for k in G.successors(j)], 'k', 'succ(j)')
         for j in G.nodes() if j not in (s, t)], 'j', 'nodes-(s,t)')

    # Set source flow at s
    P.add_constraint(
        pic.sum([x[s, j] for j in G.successors(s)], 'j', 'succ(s)') == 1)

    # Set sink flow at t
    P.add_constraint(
        pic.sum([x[i, t] for i in G.predecessors(t)], 'i', 'pred(t)') == 1)

    # Enforce edge capacities
    P.add_list_of_constraints(
        [x[e[:2]] <= e[2]['capacity'] for e in G.edges(data=True)],  # list of constraints
        [('e', 2)],     # e is a double index
        'edges')        # set the index belongs to

    # Enforce flow non-negativity
    P.add_list_of_constraints(
        [x[e] >= 0 for e in G.edges()],  # list of constraints
        [('e', 2)],  # e is a double index
        'edges')  # set the index belongs to

    # Solve
    P.set_objective('min', v)
    sol = P.solve(verbose=0, solver='cvxopt')

    # Unpack solution vector
    x = np.array(sol['cvxopt_sol']['x']).reshape(-1)[:len(G.edges())]

    return P, x


def solve_RMCFP_L_2(G, s, t, mu, delta, Gamma):
    '''
        This function solves the Robust MCFP with L_2 uncertainty set
        :param G: bidirectional graph object
        :param s: source node (0)
        :param t: sink (terminal) node (N-1)
        :param mu: nominal edges costs
        :param delta: costs deviations (uncertainty amplification)
        :param Gamma: uncertainty ball size
        :return:
        '''
    # Define problem using PICOS
    P = pic.Problem()

    # Add the flow variables and auxiliary variable
    x = {}
    for e in G.edges():
        x[e] = P.add_variable('x[{0}]'.format(e), 1)
    v = P.add_variable('v', 1)
    # Add "dummy" variables which will be exactly x in a vector form
    # to easily add the robust feasibility constraint
    x_vector = P.add_variable('x', len(G.edges))
    # Add parameter to PICOS
    mu_pic = pic.new_param('mu', mu)
    delta_pic = pic.new_param('delta', delta)
    Gamma_pic = pic.new_param('Gamma', Gamma)

    # --- Add constraints ---
    # Robust feasibility constraint (its equivalent)
    P.add_constraint(
        abs(pic.diag(Gamma_pic * delta_pic) * x_vector) <= v - (mu_pic|x_vector))

    # Enforce x_vector elements to be equal to all x variables
    P.add_list_of_constraints(
        [x_vector[i] == x[e] for (i, e) in zip(range(len(x_vector)), G.edges())])

    # Enforce flow conservation
    P.add_list_of_constraints(
        [pic.sum([x[i, j] for i in G.predecessors(j)], 'i', 'pred(j)')
         == pic.sum([x[j, k] for k in G.successors(j)], 'k', 'succ(j)')
         for j in G.nodes() if j not in (s, t)], 'j', 'nodes-(s,t)')

    # Set source flow at s
    P.add_constraint(
        pic.sum([x[s, j] for j in G.successors(s)], 'j', 'succ(s)') == 1)

    # Set sink flow at t
    P.add_constraint(
        pic.sum([x[i, t] for i in G.predecessors(t)], 'i', 'pred(t)') == 1)

    # Enforce edge capacities
    P.add_list_of_constraints(
        [x[e[:2]] <= e[2]['capacity'] for e in G.edges(data=True)],  # list of constraints
        [('e', 2)],     # e is a double index
        'edges')        # set the index belongs to

    # Enforce flow non-negativity
    P.add_list_of_constraints(
        [x[e] >= 0 for e in G.edges()],  # list of constraints
        [('e', 2)],  # e is a double index
        'edges')  # set the index belongs to

    # Solve
    P.set_objective('min', v)
    sol = P.solve(verbose=0, solver='cvxopt')

    # Unpack solution vector
    x = np.array(sol['cvxopt_sol']['x']).reshape(-1)[:len(G.edges())]

    return P, x

