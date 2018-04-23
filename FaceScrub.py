import argparse
import imageDLandCROP
import seprateDataset
import dataExtraction
import linear_regression as lr
import accuracy_compute
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
    x_train, y_train = dataExtraction.prepare_training_data_label_by_actor_order(im_data_training, [3,5], 70)
    # Add constant values for each image in x_train
    x_train = np.concatenate((x_train, np.ones([x_train.shape[0], 1])), axis=1) / 255

    # Theta initialization (1024 plus a constant theta)
    theta0 = np.ones(1025) * 0.01
    # theta0 = np.ones(1025) * 0.5
    # Train classifier
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
    Display the θs that you obtain by training using the full training dataset, and by training using a training set that contains only two images of each actor.
    :return: void
    '''
    # Get training data, validation data and testing data from part2
    im_data_training, im_data_validation, im_data_testing = part2()


    # Using full training set of Baldwin and Carell
    # 500 iterations
    theta_full, iters_full = part3(1e-5, 1e-6, 5000)
    # 5000 iterations
    # theta_full, iters_full = part3(1e-5, 1e-6, 5000)
    # 50000 iterations
    # theta_full, iters_full = part3(1e-5, 1e-6, 50000)

    # Get required actors' image validation data
    x_valid, y_valid = dataExtraction.prepare_training_data_label_by_actor_order(im_data_validation, [3, 5], 10)
    # Add constant values for each image data in x_valid
    x_valid = np.concatenate((x_valid, np.ones([x_valid.shape[0], 1])), axis=1) / 255
    # Apply hypothesis function
    y_hypothesis = lr.hypothesis(theta_full, x_valid)
    # Compute accuracy
    accuracy = accuracy_compute.accuracy(y_valid, y_hypothesis)
    print(accuracy)


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


def part5(alpha, EPS, max_iters):
    '''
    Plot the performance of the classifiers on the training and validation sets vs the size of the training set.
    :param alpha: alpha
    :param EPS: epsilon
    :param max_iters: maximum iterations
    :return: void
    '''
    i = 0
    accuracy_list = []
    no_of_images = [10, 20, 30, 40, 50, 60, 70]
    while i < 7:
        # Get training data, validation data and testing data from part2
        im_data_training, im_data_validation, im_data_testing = part2()
        # Get required actors' image training data
        # Male as 0, female as 1
        x_train, y_train = dataExtraction.prepare_training_data_label_by_gender(im_data_training, [0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0], no_of_images[i])
        # Add constant values for each image data in x_train
        x_train = np.concatenate((x_train, np.ones([x_train.shape[0], 1])), axis=1) / 255
        # Theta initialization (1024 plus a constant theta)
        theta0 = np.ones(1025) * 0.01
        # Train classifiers
        theta, costs, iters = lr.gradient_descent(lr.quadratic_cost_function, lr.derivative_quadratic_cost_function, x_train, y_train, theta0, alpha, EPS, max_iters)

        # Get required actors' image validation data
        # Male as 0, female as 1
        x_valid, y_valid = dataExtraction.prepare_training_data_label_by_gender(im_data_validation, [0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0], 10)
        # Add constant values for each image data in x_valid
        x_valid = np.concatenate((x_valid, np.ones([x_valid.shape[0], 1])), axis=1) / 255
        # Apply hypothesis function
        y_hypothesis = lr.hypothesis(theta, x_valid)
        # Compute accuracy
        accuracy = accuracy_compute.accuracy(y_valid, y_hypothesis)
        accuracy_list.append(accuracy)

        i = i + 1

    figure(1)
    plot(no_of_images, accuracy_list)
    xlabel('number of training images for each actor')
    ylabel('classifier accuracy')
    show()


def part6a():
    '''
    Show part6a handwritten functions
    :return: void
    '''
    part6a = imread("part6a.jpg")
    imshow(part6a)
    show()


def part6b():
    '''
    Show part6b handwritten functions
    :return: void
    '''
    part6b = imread("part6b.jpg")
    imshow(part6b)
    show()

def part6c():
     


# def part6d():
#     # part6d



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    switcher = {
        0: part0,
        #1: part1,
        2: part2,
        3: part3,
        4: part4,
        5: part5,
        61: part6a,
        62: part6b,
        63: part6c,
        # 64: part6d,
        #7: part7,
        #8: part8,
    }
    # Get the function from switcher dictionary
    func = switcher.get(int(args.number), lambda: "Invalid number")
    # Execute the function
    if func is part3:
        func(float(args.alpha), float(args.EPS), int(args.iteration))
    elif func is part5:
        func(float(args.alpha), float(args.EPS), int(args.iteration))
    else:
        func()
