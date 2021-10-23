# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:27:53 2021

@author: cbeltran
"""
# graficando vectores
import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.arange(-2, 2 , .2), np.arange(-2, 2, .2))
U = X
V = Y

plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                    coordinates='figure')

plt.show()
