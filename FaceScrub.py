import argparse
import imageDLandCROP
import seprateDataset
import dataExtraction
import linear_regression as lr
import numpy as np
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


def part3():
    '''
    Baldwin vs. Carell classification
    Build a classifier to distinguish pictures of Alec Baldwin from pictures of Steve Carell
    :return: void
    '''
    # Get training data, validation data and testing data from part2
    im_data_training, im_data_validation, im_data_testing = part2()
    # Split out training data and label of Baldwin and Carell
    x_train, y_train = dataExtraction.prepare_training_data(im_data_training, [3,5])

    #
    theta0 = np.ones(1024) * 0.01
    theta, cost, iter = lr.gradient_descent(lr.quadratic_cost_function, lr.derivative_quadratic_cost_function, x_train, y_train, theta0, 1e-6, 1e-5, 30000)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    switcher = {
        0: part0,
        #1: part1,
        2: part2,
        3: part3,
        #4: part4,
        #5: part5,
        #6: part6,
        #7: part7,
        #8: part8,
    }
    # Get the function from switcher dictionary
    func = switcher.get(int(args.number), lambda: "Invalid number")
    # Execute the function
    func()