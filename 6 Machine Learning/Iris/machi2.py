# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 11:12:36 2021

@author: cbeltran
"""
# Sebastian Raschka, 2015 (http://sebastianraschka.com)
# Python Machine Learning - Code Examples
#
# Chapter 2 - Training Machine Learning Algorithms for Classification
#
# S. Raschka. Python Machine Learning. Packt Publishing Ltd., 2015.
# GitHub Repo: https://github.com/rasbt/python-machine-learning-book
#
# License: MIT
# https://github.com/rasbt/python-machine-learning-book/blob/master/LICENSE.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    
    def __init__(self, eta=0.1, n_iter=8):
        # in the class initializer we set the learning rate(eta) 
        # and number of iterations to run(n_iter)
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        # pass in the X and y to fit our new perceptron model
        # initiate our numpy array of weights and array of errors
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        # train our model by looping n_iter times
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                # for each individual X and y data pair, update the
                # update our model weights via learning rate * prediction error
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        # return our model after it has passed n_iter times through our training dataset
        return self
    
    def net_input(self, X):
        # helper function for predict function
        # to get the value for weights * x + weight_0
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        # predict on unseen data with our trained model
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#############################################################################
print(50 * '=')
print('Section: Training a perceptron model on the Iris dataset')
print(50 * '-')

filename ="D:/Usuarios/JuanGuarin/Escritorio/Programming/Python/Semillero FICOMACO/6 Machine Learning/Iris/iris.data"

df = pd.read_csv(filename, header=None)
print(df.tail())

#############################################################################
print(50 * '=')
print('Plotting the Iris data')
print(50 * '-')

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)


# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
# plot data
plt.scatter(X[0:50, 0], X[0:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

#plt.tight_layout()
# plt.savefig('./images/02_06.png', dpi=300)
plt.show()

#############################################################################
print(50 * '=')
print('Training the perceptron model')
print(50 * '-')

ppn = Perceptron(eta=1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

# plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)
plt.show()

#############################################################################
print(50 * '=')
print('A function for plotting decision regions')
print(50 * '-')


def plot_decision_regions(X, y, classifier, resolution=0.01):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.5, c=cmap(idx),
                    marker=markers[idx], label=cl)


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()


