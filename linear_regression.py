from scipy.misc import imread
import numpy as np
from matplotlib.pyplot import *
from numpy.linalg import norm


def quadratic_cost_function(x, y, theta):
    '''
    Quadratic cost function
    :param x: features
    :param y: targets
    :param theta: parameter theta
    :return: cost
    '''
    return np.sum((np.dot(x, theta) - y)**2)


def derivative_quadratic_cost_function(x, y, theta):
    '''
    Derivative of quadratic cost function
    :param x: features
    :param y: targets
    :param theta: parameter theta
    :return: gradient
    '''
    return 2 * np.sum((np.dot(x, theta) - y) * x.T, axis=1)


def gradient_check(f, x, y, theta, h):
    '''
    Check the accuracy of the derivative of quadratic cost function
    :param f: cost function
    :param x: features
    :param y: targets
    :param theta: parameter theta
    :param h: tiny variance
    :return: gradient
    '''
    theta1 = theta.copy()
    theta[0] = theta[0] + h
    return (f(x, y, theta) - f(x, y, theta1)) / h


def gradient_descent(f, gradf, x, y, init_t, alpha, EPS, max_iter):
    '''
    Processing gradient descent to converge the best theta(s)
    :param f: cost function
    :param gradf: derivative of cost function
    :param x: features
    :param y: targets
    :param init_t: the initial theta(s)
    :param alpha: learning rate
    :param EPS: epsilon
    :param max_iter: maximum iteration
    :return: converged theta(s), cost, no. of iteration processed
    '''
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    iter = 0
    costs = []
    iters = []

    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * gradf(x, y, t)
        cost = f(x, y, t)
        c = gradient_check(quadratic_cost_function, x, y, t, 1e-10)
        costs.append(cost)
        iters.append(iter)
        iter += 1
    return t, costs, iters


def hypothesis(theta, x):
    '''
    Hypothesis function
    :param theta: Vector of theta values
    :param x: Matrix of image data
    :return: Classify result
    '''
    return np.dot(x, theta)
