# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 18:21:33 2021

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Filename='presion_temp.txt'

M=np.loadtxt(Filename,delimiter=' ', comments='#', skiprows=1,unpack=True, usecols=(1,5))

T=M[0,0:40]
d=M[1,0:40]

plt.plot(T,d,'b*')
plt.show()

def fun(T, T0, d0):
    return d0*(T0+273)/(T+273)

par_ini=[15, 1.225]
best_val,covar=curve_fit(fun, T, d, p0=par_ini)

temp=best_val[0]
dens=best_val[1]

print('T0 = ', temp, 'd0 = ', dens, 'Desviación estandar = ', np.sqrt(np.diag(covar)))

#ddens,dT=np.sqrt(np.diag(covar))

Temp2=np.linspace(T[0],T[-1], 500)
plt.plot(Temp2,fun(Temp2, temp, dens), 'r-')
#plt.plot(ht,fun(ht, Gmt-dGmt, R+dR), 'g-')
#plt.plot(ht,fun(ht, Gmt+dGmt, R-dR), 'g-')
plt.show()