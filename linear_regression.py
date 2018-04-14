from scipy.misc import imread
import numpy as np
from matplotlib.pyplot import *
from numpy.linalg import norm


def quadratic_cost_function(x, y, theta):
    '''
    :param x: features
    :param y: targets
    :param theta: parameter theta
    :return: cost
    '''
    # print (x.shape, y.shape, theta.shape)
    # print(np.dot(x, theta.T))
    # print(np.dot(x, theta.T) - y.T)
    return np.sum((np.dot(x, theta) - y)**2)


def derivative_quadratic_cost_function(x, y, theta):
    '''
    :param x: features
    :param y: targets
    :param theta: parameter theta
    :return: gradient
    '''
    return 2 * np.sum((np.dot(x, theta) - y) * x.T, axis=1)


def check(f, x, y, theta, h):
    theta1 = theta.copy()
    theta[0] = theta[0] + h
    return (f(x, y, theta) - f(x, y, theta1)) / h

def gradient_descent(f, gradf, x, y, init_t, alpha, EPS, max_iter):
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    iter = 0
    cost = 0

    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * gradf(x, y, t)
        cost = f(x, y, t)
        c = check(quadratic_cost_function, x, y, t, 1e-10)
        print(iter, gradf(x, y, t), c, cost)
        iter += 1
    return t, cost, iter