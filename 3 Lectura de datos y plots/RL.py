import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq #minimos cuadrados
from scipy import stats  # desde la libreria de estadistica

from sklearn import linear_model #para hacer regresion lineal
from sklearn.metrics import mean_squared_error, r2_score

path='velocity.txt'  #nombre del archivo

datos=np.loadtxt(path, dtype='float',delimiter=',',comments='#',unpack='true',skiprows=1)[:,:]
cols,fil=datos.shape
t=datos[0].reshape(-1, 1)
vy=datos[1]
plt.plot(t,vy,'*','blue')

##
#usando numpy polifit
print('usando polyfit de numpy \n')
t=datos[0]# datos originales
p=np.polyfit(t,vy,1)#polinomio interpolante de grado 1
print('m =',p[0],' b =',p[1],'\n')
y=p[0]*t+p[1]
plt.plot(t,y,'-y')
plt.show()

##
#usando sklearn
print('libreria sklearn \n')
t=datos[0].reshape(-1, 1)
regr=linear_model.LinearRegression()#libreria para hacer la regresion lineal
regr.fit(t,vy) #aplica la regresion lineal a los datos
#print('Coefficients: \n', regr.coef_)
m=regr.coef_  #pendiente
b=regr.intercept_  #intercepto
print('m =',m,' b =',b)
vy_pre=regr.predict(t)#valor predicho a partir de la regresion
print('Error cuadratico medio =',mean_squared_error(vy,vy_pre))
print('coeficiente R^2 =',r2_score(vy,vy_pre),'\n')

##
#usando minimos cuadrados (leastsq) de scipy
print('usando leastsq de scipy \n')
t=datos[0]
def dife(par,t,vy): #par->parametros de entrada
    error=vy-(par[0]*t+ par[1])
    return error
p0=[0.3,0.1] #parametros de inicio
min_cua=leastsq(dife,p0,args=(t,vy))
print('min_cua =',min_cua[0],'\n')

##
#usando la libreria stats de scipy
print('usando stats de scipy \n')
slope, intercept, r_value, p_value, std_err = stats.linregress(t,vy)
print('m =',slope,' b =',intercept)
print('R^2=',r_value**2)
