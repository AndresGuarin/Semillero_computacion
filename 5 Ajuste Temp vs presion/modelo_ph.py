# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 18:21:33 2021

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Filename='presion_temp.txt'

M=np.loadtxt(Filename,delimiter=' ', comments='#', skiprows=1,unpack=True, usecols=(0,2))

h=M[0]
P=M[1]

plt.plot(h,P,'b*')
plt.show()

def fun(h, P0, H):
    return P0*np.exp(-h/H)

par_ini=[101, 15500]
best_val,covar=curve_fit(fun, h, P, p0=par_ini)
#print('P0, H=',best_val)

P0_best=best_val[0]
H_best=best_val[1]

print('P0=', P0_best, 'H=', H_best, 'Desviación estandar=', np.sqrt(np.diag(covar)))

dP0,dH=np.sqrt(np.diag(covar))

plt.figure(1)
ht=np.linspace(h[0],h[-1], 1000)
#plt.plot(ht,fun(ht, P0_best, H_best), 'r-')
#plt.plot(ht,fun(ht, Gmt-dGmt, R+dR), 'g-')
#plt.plot(ht,fun(ht, Gmt+dGmt, R-dR), 'g-')
plt.show()

P1 = np.log(P)
def fun2(h,m,b):
    return b-m*h

par_ini2=[1/15500, 2]
best_val2,covar2=curve_fit(fun2, h, P1, p0=par_ini2)
print('m, b=',best_val)

m=best_val2[0]
b=best_val2[1]

H2 = 1/m
P0_2 = np.exp(b)

print('P0=', P0_2, 'H=', H2, 'Desviación estandar=', np.sqrt(np.diag(covar2)))
plt.plot(ht,fun(ht,P0_2,H2), 'r-')

plt.figure(2)
plt.plot(h,np.log(P),'b*')
plt.plot(ht,fun2(ht,m,b), 'r-')