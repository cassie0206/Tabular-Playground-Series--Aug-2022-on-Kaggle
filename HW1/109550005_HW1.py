import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



#----------------------------------------------------------------------------------------
# Part 1
def compute_MSE(X, y, theta):
    # calculate Mean-Squre-Error according to the formula
    return np.sum((np.dot(X, theta) - y) ** 2) / len(X)


def compute_gradient_descent(X, y, theta, alpha):
    # according to the gradient descent formula
    gradient = np.dot(X.T, (np.dot(X, theta) - y)) / len(X)
    theta = theta - alpha * gradient.astype(float) 
    
    return theta


def main_1():
    iteration = 1000
    alpha = 0.01
    x_train, x_test, y_train, y_test = np.load('regression_data.npy', allow_pickle=True)
    #plt.plot(x_train, y_train, '.', label='training data')
    #plt.plot(x_test, y_test, '.', label='testing data')

    # create the coefficient of intercept(= 1) and concatenate by column(axis=1) to do dot later
    intercept = np.ones((len(x_train), 1))
    x_train = np.concatenate((intercept, x_train), axis=1)

    # random initilaize theta as 0, 0
    theta = np.array([0, 0])
    # initialize cost list for learning curve
    loss = []
    for i in range(iteration):      
        loss.append(compute_MSE(x_train, y_train, theta))
        theta = compute_gradient_descent(x_train, y_train, theta, alpha)

    '''
    # draw the linear line
    plt.plot(x_test, y_pred, '-', label='predictive line')
    plt.legend(loc='upper left')
    plt.show()
    '''
    
    #create the coefficient of intercept(= 1) and concatenate by column(axis=1) to do dot later
    intercept = np.ones((len(x_test), 1))
    x_test = np.concatenate((intercept, x_test), axis=1)
    print("[Linear regression model]")
    print("Mean Square Error: ", compute_MSE(x_test, y_test, theta))
    print("Intercepts: ", theta[0])
    print("Weights:", theta[1])
    #----------------------------------------------------------------------------------------
    # graw the learning curve
    plt.plot(np.arange(1000), loss, '-', label='Training Data')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.show()
    

#----------------------------------------------------------------------------------------
#Part 2

iteration = 17000
alpha = 0.01

def compute_cross_entropy(X, t, theta):
    y = sigmoid(np.dot(X, theta))
    # without taking mean
    return np.sum((-1) * (t * np.log(y) + (1 - t) * np.log(1 - y))) 

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def logistic_regression(X, y, theta, alpha):
    re = sigmoid(np.dot(X, theta))
    theta = theta - alpha * (np.dot(X.T, (re - y)) / len(X))
    return theta

def main_2():
    iteration = 17000
    alpha = 0.01

    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    #plt.scatter(x_train, np.ones_like(x_train), c=y_train)
    #plt.show()

    # create the coefficient of intercept(= 1) and concatenate by column(axis=1) to do dot later
    intercept = np.ones((len(x_train), 1))
    x_train = np.concatenate((intercept, x_train), axis=1)

    # initialize
    loss = []
    theta = np.array([0, 0])

    for i in range(iteration):
        loss.append(compute_cross_entropy(x_train, y_train, theta))
        theta = logistic_regression(x_train, y_train, theta, alpha)

    intercept = np.ones((len(x_test), 1))
    x_test = np.concatenate((intercept, x_test), axis=1)
    print("[Logistic regression model]")
    print("Cross Entropy Error: ", compute_cross_entropy(x_test, y_test, theta))
    print("Intercepts: ", theta[0])
    print("Weights:", theta[1])

    #--------------------------------------------------------------------------
    # draw the learning curve
    plt.plot(np.arange(17000), loss, '-', label='Training Data')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.show()

       


if __name__ == '__main__': 
    main_1()
    main_2() 
