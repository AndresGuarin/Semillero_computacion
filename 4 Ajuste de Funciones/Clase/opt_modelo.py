# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:34:47 2019

@author: cbeltran
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit #la libreria para aproximar parametros de los datos
filename="velocity.txt"  #nombre del archivo
data=np.loadtxt(filename,dtype="float",delimiter=",",comments="#",unpack='true',skiprows=1)
#
print(data.shape)
t=data[0]
vx=data[1]
vy=data[2]
vz=data[3]

#plt.plot(t,vx,'r*-')
#plt.show()

##modelo, de testeo para los datos
def fun(t,a,f,psi): #t variable independiente, a y f parametros
    return a*np.sin(2*np.pi*f*t+psi)
###
par_ini=[0.9,1,1.57]# primer valor parametro a, segundo valor parametro w
best_val,covar=curve_fit(fun,t,vx,p0=par_ini)
print('a,f,psi= ',best_val,' covar=',covar)
am=best_val[0]
f=best_val[1]
psi=best_val[2]
print('am=',am,' f=',f,'psi=',psi)
tp=np.linspace(t[0],t[-1],1000)
plt.plot(t,vx,'ro',tp,fun(tp,am,f,psi),'b-')
plt.show()
