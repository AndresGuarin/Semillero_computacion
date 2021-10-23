# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:49:58 2021

@author: Juan Diego & JuanGuarin
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_interactions as iplt

%matplotlib auto

# valores iniciales (modifícalos bajo tu responsabilidad)
v0 = 20 ## v en m/s
y0=5
x0=0
alfa = 60 ## ángulo en grados
g = 9.81 ## Aceleración de la gravedad
## ecuaciones
v0x = v0 * np.cos(np.deg2rad(alfa))
v0y = v0 * np.sin(np.deg2rad(alfa))
t_total = (-v0y-np.sqrt((v0y**2)+2*g*y0))/(-g)
x_final = v0x * t_total+x0

## graficando trayectoria
t=np.linspace(0,t_total,100)
x=v0x*t + x0
y=(-g*(t**2))/2 + v0y*t + y0
vx=v0x
vy=v0y-g*t
plt.plot(x,y,color='red')

## graficando vectores
n=20
t=np.linspace(0,t_total,n)
x=v0x*t + x0
y=(-g*(t**2))/2 + v0y*t + y0
vx=np.zeros(n)+v0x
vy=v0y-g*t

# %%
"""
Created on Fri Mar 19 15:27:53 2021

@author: JuanGuarin
"""

%matplotlib auto

n=5
x=np.linspace(0,0,n)
y=np.linspace(0,0,n)

tt=np.linspace(0,3,n)
u=5*tt
v=10+10*tt-4.9*tt**2

w=[5 for i in range(n)]
s=10-9.8*tt 

t=np.linspace(0,3,1000)
x1=5*t
y1=10+10*t-4.9*t**2

plt.figure()
plt.title('Tiro parabólico, Graficando vectores pocisión y velocidad')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
Q = plt.quiver(x,y,u,v,color='red',scale_units='xy', scale=1, angles='xy', width=0.003)
V = plt.quiver(u,v,w,s,color='cyan',scale_units='xy', scale=2, angles='xy', width=0.003)
plt.plot(x1,y1,'-b')
plt.xlim(-1,np.max(u)+4)
plt.ylim(-15,np.max(v)+7)
plt.show()

