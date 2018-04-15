from scipy.misc import imread
import numpy as np
np.random.seed(0)


def prepare_training_data_label_by_actor_order(data, required_actor_list, training_data_size):
    '''
    Extract required image data from overall data set
    :param data: overall data set
    :param required_actor_list: required actors' indexes in actor list
    :param training_data_size: required number of images for each actor
    :return: required image features and targets
    '''
    x_train = np.ones([1,data[0][0].shape[0]])
    y_train = np.array([[1]*training_data_size])
    for i in range(len(required_actor_list)):
        if i is 0:
            x_train = np.vstack((x_train, data[required_actor_list[i]][:training_data_size]))
            x_train = np.delete(x_train, 0, axis=0)
            y_train = np.vstack((y_train, np.array([[i]*training_data_size])))
            y_train = np.delete(y_train, 0, axis=0)
        else:
            x_train = np.vstack((x_train, data[required_actor_list[i]][:training_data_size]))
            y_train = np.concatenate((y_train, np.array([[i]*training_data_size])), axis=1)
    return x_train, y_train[0]


def prepare_training_data_label_by_gender(data, required_actor_list, gender_list, training_data_size):
    '''
    Extract required image data from overall data set
    :param data: overall data set
    :param required_actor_list: required actors' indexes in actor list
    :param gender_list: provided gender list
    :param training_data_size: required number of images for each actor
    :return: required image features and targets
    '''
    x_train = np.ones([1,data[0][0].shape[0]])
    y_train = np.array([[1] * training_data_size])
    for i in range(len(required_actor_list)):
        if i is 0:
            x_train = np.vstack((x_train, data[required_actor_list[i]][:training_data_size]))
            x_train = np.delete(x_train, 0, axis=0)
            y_train = np.vstack((y_train, np.array([[gender_list[i]]*training_data_size])))
            y_train = np.delete(y_train, 0, axis=0)
        else:
            x_train = np.vstack((x_train, data[required_actor_list[i]][:training_data_size]))
            y_train = np.concatenate((y_train, np.array([[gender_list[i]]*training_data_size])), axis=1)
    return x_train, y_train[0]