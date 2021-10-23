# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:34:47 2019

@author: cbeltran
"""
import matplotlib.pyplot as plt
import numpy as np

filename='C:/Users/USUARIO/Desktop/Python/Semillero_Ficomaco/Graficas/velocity.txt'  #nombre del archivo

x,y1=np.loadtxt(filename, delimiter=',',usecols=(0,1),unpack=True,skiprows=1)
print('y1=',y1)
print('x=',x)
plt.plot(x,y1)
plt.show()
