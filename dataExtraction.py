from scipy.misc import imread
import numpy as np
np.random.seed(0)

def prepare_training_data(data, required_actor_list):
    x_train = np.ones([1,data[0][0].shape[0]])
    y_train = np.array([[1]*len(data[0])])
    for i in range(len(required_actor_list)):
        if i is 0:
            x_train = np.vstack((x_train, data[required_actor_list[i]])) / 255
            x_train = np.delete(x_train, 0, axis=0)
            y_train = np.vstack((y_train, np.array([[i]*len(data[required_actor_list[i]])])))
            y_train = np.delete(y_train, 0, axis=0)
        else:
            x_train = np.vstack((x_train, data[required_actor_list[i]])) / 255
            y_train = np.concatenate((y_train, np.array([[i]*len(data[required_actor_list[i]])])), axis=1)
    return x_train, y_train[0]