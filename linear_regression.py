from scipy.misc import imread
import numpy as np
from matplotlib.pyplot import *
from numpy.linalg import norm


def quadratic_cost_function(x, y, theta):
    '''
    Quadratic cost function
    :param x: features. m * 1025
    :param y: targets. m * 1
    :param theta: parameter theta. 1025 * 1
    :return: cost. 1 * 1
    '''
    return np.sum((np.dot(x, theta) - y)**2)


def derivative_quadratic_cost_function(x, y, theta):
    '''
    Derivative of quadratic cost function
    :param x: features. m * 1025
    :param y: targets. m * 1
    :param theta: parameter theta. 1025 * 1
    :return: gradient. 1025 * 1
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


def new_quadratic_cost_function(x, y, theta):
    '''
    New quadratic cost function
    :param x: features. m * 1025
    :param y: targets. m * k
    :param theta: parameter theta. 1025 * k
    :param k: the number of possible labels
    :return: cost. 1 * 1
    '''
    return np.sum((np.dot(x, theta) - y) ** 2)


def new_derivative_quadratic_cost_function(x, y, theta):
    '''
    Derivative of quadratic cost function
    :param x: features. m * 1025
    :param y: targets. m * k
    :param theta: parameter theta. 1025 * k
    :return: gradient. 1025 * k
    '''
    return (2 * np.dot((np.dot(x, theta) - y).T, x)).T


def finite_difference(f, x, y, p, q, theta, h):
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
    theta[p, q] = theta[p, q] + h
    return (f(x, y, theta) - f(x, y, theta1)) / h