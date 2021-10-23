# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:49:58 2021

@author: Juan Diego & JuanGuarin
"""

import numpy as np
import matplotlib.pyplot as plt
import animatplot as amp

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

X, Y = np.linspace(0,0,n), np.linspace(0,0,n)
U = x
V = y

block=amp.blocks.Quiver(X,Y,U,V,scale_units='xy',angles='xy',scale=1,color='blue',width=0.003)
block2=amp.blocks.Quiver(U,V,vx,vy,scale_units='xy',angles='xy',scale=5,color='cyan',width=0.003)
anim=amp.Animation([block,block2])
plt.show()