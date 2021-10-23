import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#
N=501
x=np.linspace(-10,10,N)
h=x[1]-x[0]
Id=np.identity(N-2)
Id_s=np.diag(np.diag(Id[:-1]),k=1)
Id_i=np.diag(np.diag(Id[:-1]),k=-1)
T=(-2*Id+Id_s+Id_i)/h**2
xx=x[1:-1]
U_pot=5*xx**2
Ha=-T+np.diag(U_pot)
ev,fonda=LA.eig(Ha)
idx = np.argsort(ev)  #ordena de menor a mayor
ev_s= ev[idx]
#print(ev_s)
fonda_s = fonda[:,idx]
for i in range(6):
   plt.plot(xx,10*fonda_s[:,i]+ev_s[i],xx,U_pot,'-r')
   plt.axhline(ev_s[i],xx[0],xx[-1],c='k',ls='--')
plt.xlim(-3,3)
plt.ylim(-0.5,30)
plt.show()

