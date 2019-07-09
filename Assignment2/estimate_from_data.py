
import numpy as np
from scipy.stats import truncnorm


def draw_xi(N):
    '''

    :param N: number of data points to draw
    :return: Xi - array of size N x 10
    '''

    # initialize Xi array
    Xi = np.zeros((N, 10))
    # vector fill data point
    Xi[:, 0] = np.random.uniform(3.5, 4.5, N)
    Xi[:, 1] = np.random.uniform(8., 10., N)
    Xi[:, 2] = np.random.uniform(6., 8., N)
    Xi[:, 3] = np.random.uniform(9., 11., N)
    Xi[:, 4] = np.random.uniform(.8, 1.2, N)
    Xi[:, 5] = np.random.uniform(.8, 1.2, N)
    Xi[:, 6] = np.random.uniform(2.5, 3.5, N)
    Xi[:, 7] = np.random.uniform(36., 44., N)
    # truncated normal distributions
    lower1, upper1 = 5700, 6300
    lower2, upper2 = 3850, 4150
    mu1, sigma1 = 6000, np.sqrt(100)
    mu2, sigma2 = 4000, np.sqrt(50)
    h_tilde1 = truncnorm((lower1 - mu1) / sigma1, (upper1 - mu1) / sigma1,
                         loc=mu1, scale=sigma1)
    h_tilde2 = truncnorm((lower2 - mu2) / sigma2, (upper2 - mu2) / sigma2,
                         loc=mu2, scale=sigma2)

    Xi[:, 8] = h_tilde1.rvs(N)
    Xi[:, 9] = h_tilde2.rvs(N)
    return Xi


def empirical_mean(Xi):
    '''

    :param Xi: array of size N x 10
    :return: mu - empirical mean
    '''

    return Xi.mean(axis=0)


def empirical_cov(Xi):
    '''

    :param Xi: array of size N x 10
    :return: Sigma - empirical covariance
    '''

    n = Xi.shape[0]
    mu = empirical_mean(Xi)
    Sigma = (1 / (n-1)) * (Xi - mu).T @ (Xi - mu)
    return Sigma


def empirical_t(Xi, g):
    '''

    :param Xi: array of size N x 10
    :param g: array of size K x 10 (in our case - K = 2 ** 10 - 1)
    :return:
    '''
    t = []  # initialize list of t_i
    mu = empirical_mean(Xi)
    for g_i in g:
        t_i = np.mean(((Xi - mu) @ g_i) ** 2)
        t.append(t_i)
    t = np.vstack(t).reshape(-1)
    return t

