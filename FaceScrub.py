import argparse
import imageDLandCROP
import seprateDataset
import dataExtraction
import linear_regression as lr
import numpy as np
from matplotlib.pyplot import *
from scipy.misc import imread
actor = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


def get_parser():
    '''
    Get command line input arguments
    :return: parser
    '''
    # Get parser for command line arguments.
    parser = argparse.ArgumentParser(description="Face Scrub")
    parser.add_argument("-n",
                        "--number",
                        dest="number")
    parser.add_argument("-a",
                        "--alpha",
                        dest="alpha")
    parser.add_argument("-e",
                        "--EPS",
                        dest="EPS")
    parser.add_argument("-i",
                        "--iteration",
                        dest="iteration")
    return parser


def part0():
    '''
    Data preparation
    To run the file, you need an empty uncropped and an empty cropped folder
    :return: void
    '''
    imageDLandCROP.DLandCROP(actor)


def part2():
    '''
    Split Dataset
    Split the data into training, validation and test sets
    :return: im_data_training, im_data_validation, im_data_testing
    '''
    im_data = seprateDataset.read(actor, 'cropped/')
    im_data_training, im_data_validation, im_data_testing = seprateDataset.split(actor, im_data, 70, 10, 10)
    return im_data_training, im_data_validation, im_data_testing


def part3(alpha, EPS, max_iters):
    '''
    Baldwin vs. Carell classification
    Build a classifier to distinguish pictures of Alec Baldwin from pictures of Steve Carell
    :return: theta
    '''
    # Get training data, validation data and testing data from part2
    im_data_training, im_data_validation, im_data_testing = part2()
    # Split out training data and label of Baldwin and Carell
    x_train, y_train = dataExtraction.prepare_training_data(im_data_training, [3,5])
    # Add constant values for each image in x_train
    x_train = np.concatenate((x_train, np.ones([x_train.shape[0], 1])), axis=1) / 255

    # Theta initialization (1024 plus a constant theta)
    theta0 = np.ones(1025) * 0.01
    # theta0 = np.ones(1025) * 0.5
    theta, costs, iters = lr.gradient_descent(lr.quadratic_cost_function, lr.derivative_quadratic_cost_function, x_train, y_train, theta0, alpha, EPS, max_iters)
    return theta, iters

    # Plot the relationship between costs and iterations to see if there are any overfit, if the alpha is too large.
    # figure(1)
    # plot(iters, costs)
    # xlabel('iters')
    # ylabel('costs')
    # show()
    # print(costs)
    # print(iters)


def part4():
    '''

    :return:
    '''
    # Using full training set of Baldwin and Carell
    # 500 iterations
    theta_full, iters_full = part3(1e-5, 1e-6, 500)
    # 5000 iterations
    # theta_full, iters_full = part3(1e-5, 1e-6, 5000)
    # 50000 iterations
    # theta_full, iters_full = part3(1e-5, 1e-6, 50000)

    # Using two images of Baldwin and Carell
    # Get training data, validation data and testing data from part2
    im_data_training, im_data_validation, im_data_testing = part2()
    y_train = np.array([1, 1, 0, 0])
    x_train = im_data_training[3][10]
    x_train = np.vstack((x_train, im_data_training[3][11]))
    x_train = np.vstack((x_train, im_data_training[5][20]))
    x_train = np.vstack((x_train, im_data_training[5][21]))
    # Add constant values for each image in x_train
    x_train = np.concatenate((x_train, np.ones([x_train.shape[0], 1])), axis=1) / 255
    # Theta initialization
    theta0 = np.ones(1025) * 0.01
    theta_2, costs, iters_2 = lr.gradient_descent(lr.quadratic_cost_function, lr.derivative_quadratic_cost_function, x_train, y_train, theta0, 1e-5, 1e-6, 50)

    # Show image of theta
    new_theta_full = np.reshape(theta_full[:1024], (32, 32))
    imshow(new_theta_full, cmap="RdBu", interpolation="spline16")
    show()
    new_theta_2 = np.reshape(theta_2[:1024], (32, 32))
    imshow(new_theta_2, cmap="RdBu", interpolation="spline16")
    show()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    switcher = {
        0: part0,
        #1: part1,
        2: part2,
        3: part3,
        4: part4,
        #5: part5,
        #6: part6,
        #7: part7,
        #8: part8,
    }
    # Get the function from switcher dictionary
    func = switcher.get(int(args.number), lambda: "Invalid number")
    # Execute the function
    if func is part3:
        func(float(args.alpha), float(args.EPS), int(args.iteration))
    else:
        func()