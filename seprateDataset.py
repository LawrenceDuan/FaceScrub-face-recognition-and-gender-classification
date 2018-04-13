import os
from scipy.misc import imread
import numpy as np


def read(actor, directory_string):
    '''
    Read in image stored in local directory
    :param actor: list of actors' names want
    :param directory_string: the string of the local directory
    :return: image data
    '''
    directory = os.fsencode(directory_string)
    actor_list = actor

    im_data = []
    for i in range(len(actor_list)):
        im_data.append([])

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".DS_Store"):
            im = imread(directory_string + filename, 'L')
            for i in range(len(actor_list)):
                actor = actor_list[i]
                actor_lastName = actor.split(' ')[1].lower()
                if actor_lastName in filename:
                    im_data[i].append(im.flatten())
    return im_data


def split(actor, im_data, train_num, valid_num, test_num):
    '''
    Split all image data into user set training set, validation set and testing set
    :param actor: list of actors' names
    :param im_data: image data
    :param train_num: no. of training data
    :param valid_num: no. of validaiton data
    :param test_num: no. of testing data
    :return:
    '''
    im_data_training = []
    im_data_validation = []
    im_data_testing = []
    for i in range(len(actor)):
        im_data_training.append([])
        im_data_validation.append([])
        im_data_testing.append([])

    for i in range(len(im_data)):
        random_index_list = np.random.choice(np.arange(len(im_data[i])), train_num + valid_num + test_num, replace=False)

        for j in range(len(random_index_list)):
            if j < train_num:
                im_data_training[i].append(im_data[i][random_index_list[j]])
            elif j < train_num + valid_num:
                im_data_validation[i].append(im_data[i][random_index_list[j]])
            else:
                im_data_testing[i].append(im_data[i][random_index_list[j]])
    return im_data_training, im_data_validation, im_data_testing