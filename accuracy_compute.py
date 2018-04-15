

def accuracy(y_valid, y_hypothesis):
    '''
    Compute classifier accuracy.
    :param y_valid: validation set labels.
    :param y_hypothesis: hypothesis set labels
    :return: accuracy in %
    '''
    # Compute accuracy
    correct_count = 0
    for i in range(len(y_valid)):
        if y_hypothesis[i] <= 0.5 and y_valid[i] == 0: correct_count = correct_count + 1
        if y_hypothesis[i] > 0.5 and y_valid[i] == 1:
            correct_count = correct_count + 1
    accuracy = correct_count / len(y_valid) * 100
    # print(y_valid)
    # print(y_hypothesis)
    return accuracy