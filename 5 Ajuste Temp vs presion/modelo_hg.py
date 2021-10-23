# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 21:39:20 2021

@author: personal
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


Filename='presion_temp.txt'

M=np.loadtxt(Filename,delimiter=' ', comments='#', skiprows=1,unpack=True, usecols=(0,3))


h=M[0]
g=M[1]

G=6.67*(10**-11)
mt=6*(10**24)
#r=6.4*(10**6)


plt.plot(h,g,'b*')
plt.show()

def fun(h, a, r):
    return (a)/((r+h)**2)

par_ini=[G*mt, 6.4*(10**6)]
best_val,covar=curve_fit(fun, h, g, p0=par_ini)
print('G*mt, r=',best_val)

Gmt=best_val[0]
R=best_val[1]

print('G*mt=', Gmt, 'R=', R, 'Desviación estandar=', np.sqrt(np.diag(covar)))

#RL usando polyfit
#p=np.polyfit(t,v,1)
#print('m=',p[0], 'b=',p[1], '\n')
#y=p[0]*t+p[1]
#plt.plot(t,y, 'y')
dGmt,dR=np.sqrt(np.diag(covar))

ht=np.linspace(h[0],h[-1], 1000)
plt.plot(ht,fun(ht, Gmt, R), 'r-')
#plt.plot(ht,fun(ht, Gmt-dGmt, R+dR), 'g-')
#plt.plot(ht,fun(ht, Gmt+dGmt, R-dR), 'g-')
plt.show()