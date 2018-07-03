'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt, exp
# from nnScript import sigmoid, nnObjFunction, nnPredict 

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
#def sigmoid(z):
# Replace this with your nnObjFunction implementation
#def nnObjFunction(params, *args):
    
# Replace this with your nnPredict implementation
#def nnPredict(w1,w2,data):

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-z))

iteration = 0
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
    print("iteration: ", iteration)
    iteration += 1

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    yl = []
    for l in training_label:
        y = np.zeros(int(np.max(training_label)) + 1)
        y[int(l)] = 1
        yl.append(y)
    yl = np.array(yl)

    n = len(yl)

    # print("Starting hidden node value calculations\n")
    z = []
    z_no_hidden = []
    for image in training_data:
        z_i = sigmoid(np.dot(w1, np.append(image, 1)))
        # add hiden bias
        z_no_hidden.append(z_i)
        z_i = np.append(z_i, 1)
        z.append(z_i)
    z = np.array(z)
    z_no_hidden = np.array(z_no_hidden)
    # print("hidden node values complete")

    out = []
    for node in z:
        out_j = sigmoid(np.dot(w2, node))
        out.append(out_j)
    out = np.array(out)
    # print("output values complete")


    # error function objective val
    for i, y_i in enumerate(yl):
        for l, y_i_l in enumerate(y_i):
            obj_val += y_i_l * np.log(out[i][l]) + (1 - y_i_l) * np.log(1 - out[i][l])
    obj_val = (-1 / n) * obj_val

    # regularization objective val
    reg_obj_val = 0
    for w1j in w1:
        for w1jp in w1j:
            reg_obj_val += w1jp ** 2
    for w2l in w2:
        for w2lj in w2l:
            reg_obj_val += w2lj ** 2
    reg_obj_val = ( lambdaval / (2 * n) ) * reg_obj_val

    obj_val += reg_obj_val
    

    # Gradient descent:
    djw2 = np.matmul((out - yl).T, z)

    w2_no_hidden = np.delete(w2, len(w2[0]) - 1, 1)
    mat_1 = np.multiply((1 - z_no_hidden), z_no_hidden)
    delta_x_w1 = np.matmul((out - yl), w2_no_hidden)
    mat_1_x_mat_2 = np.multiply(mat_1, delta_x_w1)
    djw1 = np.matmul(mat_1_x_mat_2.T, np.hstack((training_data,np.ones((training_data.shape[0],1)))))

    # regularization:
    reg_djw2 = (1 / n) * (djw2 + lambdaval * w2)
    reg_djw1 = (1 / n) * (djw1 + lambdaval * w1)

    obj_grad = np.concatenate((reg_djw1.flatten(), reg_djw2.flatten()), 0) 
    
    # print("regularizations complete)
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



# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
