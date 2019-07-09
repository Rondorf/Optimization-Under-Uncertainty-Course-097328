
import numpy as np
import picos as pic

MAX_MAN_HOUR = 3000


def get_binary_vectors(n):
    '''
    this function returns all the binary vectors in length 10 excluding zero
    :param n: number of bits
    :return: g - array of size (2 ** n)-1 x n (all binary vectors excluding zero)
    '''
    g = []
    for dec in range(1, 2 ** n):
        bin = "{0:b}".format(dec)  # convert decimal to binary
        bin_list = [int(d) for d in str(bin)]  # convert to binary digits
        if len(bin) < n:  # for all lists to be the same size
            bin_list = [0] * (n - len(bin)) + bin_list
        # append to the power set
        g.append(bin_list)
    g = np.vstack(g)
    return g


def define_art_params(c, q, xi_0):
    x_dim = len(c)
    xi_dim = len(xi_0)

    E = np.concatenate((np.eye(x_dim), np.zeros((1, x_dim))), axis=0)
    c_tilde = E @ c
    # c_tilde = pic.new_param('c_tilde', E @ c)
    # E = pic.new_param('E', E)

    # Compute E_tilde matrices
    E_tilde_1 = np.zeros((x_dim + 1, xi_dim))
    E_tilde_1[:x_dim, :x_dim] = q[0] * np.eye(x_dim)
    E_tilde_1[-1, -2] = q[0]

    E_tilde_2 = np.zeros((x_dim + 1, xi_dim))
    E_tilde_2[:x_dim, x_dim:2*x_dim] = q[1] * np.eye(x_dim)
    E_tilde_2[-1, -1] = q[1]

    E_tilde_3 = E_tilde_1 + E_tilde_2

    E_tilde_4 = np.zeros((x_dim + 1, xi_dim))

    E_tilde_arr = [E_tilde_1,
                   E_tilde_2,
                   E_tilde_3,
                   E_tilde_4]
    # E_tilde_arr = [pic.new_param('E_tilde_1', E_tilde_1),
    #                pic.new_param('E_tilde_2', E_tilde_2),
    #                pic.new_param('E_tilde_3', E_tilde_3),
    #                pic.new_param('E_tilde_4', E_tilde_4)]
    # Compute E matrices
    E_1 = E_tilde_1 / q[0]
    E_2 = E_tilde_2 / q[1]

    E_arr = [E_1, E_2]
    # E_arr = [pic.new_param('E_1', E_1), pic.new_param('E_2', E_2)]

    e5 = np.zeros(x_dim + 1)    # e5 since for this case x_dim=4
    e5[-1] = 1.
    # e5 = pic.new_param('e5', e5)
    return E, c_tilde, E_tilde_arr, E_arr, e5


def unpack_params_picos(params):
    '''

    :param params: list [c, q, xi_0, b]
    :return:
    '''
    c = pic.new_param('c', params[0])
    q = pic.new_param('q', params[1])
    xi_0 = pic.new_param('xi_0', params[2])
    b = pic.new_param('b', params[3])
    B = pic.new_param('B', np.diag(params[3]))
    return c, q, xi_0, b, B


def convert_art_params_picos(E, c_tilde, E_tilde_arr, E_arr, e5):
    E = pic.new_param('E', E)
    c_tilde = pic.new_param('c_tilde', c_tilde)
    E_tilde_arr = [pic.new_param('E_tilde_1', E_tilde_arr[0]),
                   pic.new_param('E_tilde_2', E_tilde_arr[1]),
                   pic.new_param('E_tilde_3', E_tilde_arr[2]),
                   pic.new_param('E_tilde_4', E_tilde_arr[3])]
    E_arr = [pic.new_param('E_1', E_arr[0]), pic.new_param('E_2', E_arr[1])]
    e5 = pic.new_param('e5', e5)
    return E, c_tilde, E_tilde_arr, E_arr, e5


def is_feasible(x, Xi, c, q, xi_0):
    x_tilde = np.concatenate((x, [-1]))
    feasible = []
    _, _, _, E_arr, _ = define_art_params(c, q, xi_0)
    for xi_i in Xi:
        if (E_arr[0] @ xi_i).T @ x_tilde <= MAX_MAN_HOUR \
                and (E_arr[1] @ xi_i).T @ x_tilde <= MAX_MAN_HOUR \
                and np.all(x >= 0):
            feasible.append(True)
        else:
            feasible.append(False)
    return sum(feasible) * 100 / len(feasible), feasible


def average_profit(x, Xi, c, q, xi_0):
    x_tilde = np.concatenate((x, [-1]))
    profit = []
    _, c_tilde, E_tilde_arr, _, _ = define_art_params(c, q, xi_0)
    for xi_i in Xi:
        profit.append(-max((E_tilde_arr[0] @ xi_i).T @ x_tilde,
                           (E_tilde_arr[1] @ xi_i).T @ x_tilde,
                           (E_tilde_arr[2] @ xi_i).T @ x_tilde,
                           (E_tilde_arr[3] @ xi_i).T @ x_tilde) + c_tilde @ x_tilde)
    return np.mean(profit)



