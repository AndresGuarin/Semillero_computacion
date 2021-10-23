 #Ejemplos de uso de librerias para el analisis de datos  para interpolaciones
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # interpolación
from scipy.optimize import curve_fit  # optimizacion
#
#interpolación de datos
x = np.array([2.,3.,4.,5.,6.])
y = np.array([2.,6.,5.,5.,6.])
#
xnew=np.linspace(2,6,60)
f=interp1d(x,y,kind='cubic') # cambie kind entre 'linear','quadratic' o 'cubic'
plt.plot(x,y,'ro',xnew,f(xnew),'g-')
plt.axis([1.5,6.5,1.5,6.5]);
plt.show()
#
#curvigrafos
filename="tabla.txt"  #nombre del archivo
data=np.loadtxt(filename,dtype="float",delimiter=",",unpack="True")
col,row=data.shape  #tamaño del arreglo, numero de columnas y filas
print(row,col)
Temperature=data[0]
Pressure=data[1]
#plt.plot(Temperature,Pressure,'ro')
#plt.xlabel('T($^oC$)')
#plt.ylabel('Presion (Pa)')
#plt.show()

def poli(X,A,B,C,D):

  return  A*(X**3) + B*(X**2) + C*X + D

best_vals, covar = curve_fit(poli, Temperature,Pressure)
A=best_vals[0]
B=best_vals[1]
C=best_vals[2]
D=best_vals[3]
#
print("El coeficiente A es= ", best_vals[0], ' ; El coeficiente B es= ', best_vals[1], " ; El coeficiente C es: ", best_vals[2], 'El coeficiente D es= ', best_vals[3])
plt.plot(Temperature,Pressure, 'ro', label='Datos O.') #Datos originales de la tabla.
plt.plot(Temperature,poli(Temperature,A,B,C,D), "g--", label='Optimización') #Arreglos con la optimización.

plt.legend(loc='upper left')
plt.xlabel('Temperatura ($^o$C)')
plt.ylabel('Presión (kPa)')
plt.show()
