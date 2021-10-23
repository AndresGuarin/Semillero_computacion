"""
Created on Fri Apr 23 17:48:57 2021

@author: cbeltran
"""
import numpy as np
import matplotlib.pyplot as plt

#plt.figure()
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family':'serif', 'serif':['Times']})
#plt.rc('font', size=14.0)
t=np.linspace(10,45,100)
#plt.xlabel(r'$Temperatura (°C)$')
#plt.ylabel(r'$Velocidad (ms^{-1})$')
v=331.321*np.sqrt(1+t/273.15)
plt.plot(t,v)
#plt.savefig('velsonido.png', bbox_inches='tight')
#plt.show()
