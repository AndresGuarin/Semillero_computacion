# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 18:17:12 2021

@author: SamuelRosado, Juan Guarin
"""

# In[1]: Importación de librerías
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

Direccion = 'D:\\Usuarios\\JuanGuarin\\Escritorio\\Programming\\Python\\Semillero FICOMACO\\5 Ajuste Temp vs presion\\presion_temp.txt'

M=np.loadtxt(Direccion, delimiter=' ', comments='#', skiprows=1,unpack=True, usecols=(1,4))

# h =
# Temp = 
# Pr = 
# g = 
# v_S = 
# dens = 
# visc = 
# condc = 
t=M[0]
v=M[1]


plt.plot(t,v,'*')
#plt.show()

def fun(t,v0,T0):
    return v0*np.sqrt(1+(t/T0))

par_ini=[340,273]
best_val,covar=curve_fit(fun, t, v, p0=par_ini)
print('v0,T0=',best_val)

vel=best_val[0]
temp=best_val[1]

print('vel=',vel, 'temp=', temp, 'covar=', covar)
t=np.linspace(t[0],t[-1],250)
plt.plot(t,fun(t, vel, temp))

#RL usando polyfit
#p=np.polyfit(t,v,1)
#print('m=',p[0], 'b=',p[1], '\n')
#y=p[0]*t+p[1]
#plt.plot(t,y, 'y')


vt=np.linspace(t[0],t[-1], 1000)
#plt.plot(vt,fun(vt,vel,temp), 'r-')
plt.show()