import numpy as np

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-z))

iteration = 0
yl = None
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    global iteration
    # print("iteration: ", iteration)
    iteration += 1

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    global yl
    if not isinstance(yl, np.ndarray):
        yl = []
        for l in training_label:
            y = np.zeros(int(np.max(training_label)) + 1)
            y[int(l)] = 1
            yl.append(y)
        yl = np.array(yl)

    n = len(yl)

    # print("Starting hidden node value calculations")

    train_data_with_bias = np.hstack((training_data, np.ones((training_data.shape[0],1))))
    z_no_bias = sigmoid(np.matmul(train_data_with_bias, w1.T))
    z = np.hstack((z_no_bias, np.ones((z_no_bias.shape[0],1))))
    # print("hidden node values complete")

    out = sigmoid(np.matmul(z, w2.T))
    # print("output values complete")

    obj_val = (-1 / n) * np.sum(np.multiply(yl, np.log(out)) + np.multiply((1 - yl), np.log(1 - out)))
    print("iteration: ", iteration, " obj val:", obj_val)

    reg_obj_val = (lambdaval / (2 * n)) * (np.sum(w1**2) + np.sum(w2**2))
    obj_val += reg_obj_val
    # print("regularization obj val complete")

    # Gradient descent:
    djw2 = np.matmul((out - yl).T, z)

    # w2_no_hidden = np.delete(w2, len(w2[0]) - 1, 1)
    w2_no_hidden = np.delete(w2, n_hidden - 1, 1)
    mat_1 = np.multiply((1 - z_no_bias), z_no_bias)
    delta_x_w1 = np.matmul((out - yl), w2_no_hidden)
    mat_1_x_mat_2 = np.multiply(mat_1, delta_x_w1)
    djw1 = np.matmul(mat_1_x_mat_2.T, train_data_with_bias)
    # print("gradient descent complete")

    # regularization:
    reg_djw2 = (1 / n) * (djw2 + lambdaval * w2)
    reg_djw1 = (1 / n) * (djw1 + lambdaval * w1)

    obj_grad = np.concatenate((reg_djw1.flatten(), reg_djw2.flatten()), 0) 
    
    # print("regularized gradient descent complete")
    #
    #
    #
    #
    #

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    for image in data:
        z = sigmoid(np.dot(w1, np.append(image, 1)))
        prediction_labels = sigmoid(np.dot(w2, np.append(z, 1)))
        labels = np.append(labels, np.argmax(prediction_labels))

    return labels
