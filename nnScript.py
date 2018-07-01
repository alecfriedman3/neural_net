import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt, exp


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # if isinstance(z, int) or isinstance(z, float):
    #     return 1 / (1 + exp(-z))
    # Z = z[:]
    # for y in range(0,len(z)):
    #     Z[y] = sigmoid(z[y])
    # return Z
    return 1.0 / (1.0 + np.exp(-z))

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    redundant_features_counts = np.zeros(784)
    for image in test_data:
        for j, feature_value in enumerate(image):
            if feature_value == 0:
                redundant_features_counts[j] += 1

    wanted_features = [f for f in range(len(redundant_features_counts)) if redundant_features_counts[f] != len(test_data)]
    def filterRedundancy(data):
        new_data = np.array(list(map(lambda image: image[wanted_features], data)))
        return new_data
    
    train_data = filterRedundancy(train_data)
    validation_data = filterRedundancy(validation_data)
    test_data = filterRedundancy(test_data)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

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
    # print("running nnobjfunc with weights: \n\nlenghts = ",len(w1),len(w2), len(w1[0]), len(w2[0]),"\n\nw1 = ", w1,"\n\nw2 = ", w2)
    # print(training_label)

    yl = []
    for l in training_label:
        y = np.zeros(10)
        y[int(l)] = 1
        yl.append(y)
    yl = np.array(yl)

    n = len(yl)

    # for image in training_data:
    # print("labels should be:", training_label)
    # prediction = nnPredict(w1, w2, training_data)
    # print("PREDICTION, THIS JUST IN!", prediction)
    print("Starting hidden node value calculations\n")
    z = []
    z_no_hidden = []
    for image in training_data:
        # z_i = sigmoid(feedForwardSummation(w1, np.append(image, 1)))
        z_i = sigmoid(np.dot(w1, np.append(image, 1)))
        # add hiden bias
        # print("zi value here before bias: ", z_i)
        z_no_hidden.append(z_i)
        z_i = np.append(z_i, 1)
        # print("zi value here: ", z_i)
        z.append(z_i)
    z = np.array(z)
    z_no_hidden = np.array(z_no_hidden)
    # print("hidden node values: ", z)
    print("hidden node values complete")

    out = []
    for node in z:
        # out_j = sigmoid(feedForwardSummation(w2, node))
        out_j = sigmoid(np.dot(w2, node))
        out.append(out_j)
    out = np.array(out)
    # print("output values: ", out)
    print("output values complete")


    # error function objective val
    for i, y_i in enumerate(yl):
        for l, y_i_l in enumerate(y_i):
            # (yil ln oil + (1 − yil) ln(1 − oil))
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
    # print("regularizations: ", reg_djw1, reg_djw2)
    reg_djw1.flatten()
    obj_grad = np.concatenate((reg_djw1.flatten(), reg_djw2.flatten()), 0) 
    
    # print("regularizations complete, obj_val:", obj_val, "gradient w1: ", reg_djw1, "gradient w2: ", reg_djw2)
    # print(djw1, djw2)

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
        # prediction_labels = feedForwardPropogation(w1, w2, image)
        # print("PREDICTION----------------------------------------------------\n\n", prediction_labels, "\n\n")
        labels = np.append(labels, np.argmax(prediction_labels))
        # print("LABELS SO FAR!", labels)

    return labels

# def feedForwardPropogation(w1, w2, image):
#     z = sigmoid(feedForwardSummation(w1, np.append(image, 1)))
#     labels = sigmoid(feedForwardSummation(w2, np.append(z, 1)))
#     return labels

def feedForwardSummation(weight, input_vector):
    a = [0] * (len(weight))
    for i, weight_i_vector in enumerate(weight):
        for j, weight_i_j in enumerate(weight_i_vector):
            a[i] += weight_i_j * input_vector[j]
    return a

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')