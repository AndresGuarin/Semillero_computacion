import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, splev, splrep

dirvel='C:/Users/UIS/Desktop/Python/Semillero_Ficomaco/Graficas/velocity.txt'

datos=np.loadtxt(dirvel,dtype='float',delimiter=',',unpack='True',skiprows=1)[:, :]
col,fil=datos.shape

print(datos.shape)
print(datos)
#
#plt.plot(t,vx,'r',t,vy,'b',t,vz,'g')
#plt.plot(datos[0,0:fil],datos[1,0:fil],'*') #marcador de estrella
#plt.show()
t=datos[0,0:fil]
p=datos[1,0:fil]

#interpoalcion lagrage  
def lag(tp):
      lag_p=lagrange(t,p)
      return lag_p(tp)

print(lag(320))

#evaluacion de la funcion usando splines
def fk(tp):
    tck=splrep(t,p,s=0)
    return splev(tp,tck)

print(fk(0.043))

# Graficando interpolaciones
tt=np.linspace(0,0.1,100)
sp=np.zeros(100)
lg=np.zeros(100)

for i in range(100):
    sp[i]=fk(tt[i])
    lg[i]=lag(tt[i])

plt.figure(1)
plt.plot(tt,sp,'r')
#plt.plot(tt,lg,'cyan')
plt.plot(t,p,'b')
plt.show()
