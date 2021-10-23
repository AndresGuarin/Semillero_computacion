# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:28:52 2021

@author: cbeltran
"""
# manejo de vectores en python
import numpy as np #libreria para el manejo de vectores y matrices
#
#constructor basico
a=np.arange(3)
print ('a=',a) #muestra el contenido de la variable a
print ('a(1)=',a[1])#los indices empiezan en cero
b=a.reshape(3,1)
print ('b=',b) # vector columna
print ('b(0)=',b[0])
b[0]=3
print('b=',b)
#
# vector de unos
v=np.ones(5)
print ('v=',v)
#
# vector de ceros
o=np.zeros(5) 
print ('o=',o)
#
# numeros aleatorios
va=np.random.normal(size=5)
vb=np.random.uniform(size=5)
print('va=',va)
print('vb=',vb)
#
vc=np.random.randint(low=1, high=10, size=5)
print('vc=',vc)
#
#multiplicación por un escalar
va=10*va
# sumar todos los elementos por un número
vb=1+vb
print('va=',va)
print('vb=',vb)
#
#producto cruz
w=np.array([1,0,0])
y=np.array([0,1,0])
print('wxy=',np.cross(w,y))
#
#suma de vectores
vc=va+vb
print('vc=',vc)
#
#producto escalar de vectores
x=np.dot(va,vb)
print('x=',x)
print('x_=',va@vb)
#
# norma del vector
nx=np.linalg.norm(va) #linalg=paquete de algebra lineal
print('nx=',nx)
#
nx=np.sqrt(np.dot(va,va))
print('nx_=',nx)
#
#operaciones sobre los elemento de un vector
aa=np.array([1,2,3,4,5],dtype='int') #cambiar int por single, double
#
print('suma=',np.sum(aa)) #sumatoria de los elementos de array
print('aa=',aa*aa ) #producto de cada elemento por si mismo
#
bb=np.sqrt(aa)
print('bb=',bb)
#
print ('exp(bb)=',np.exp(bb)) #exponencial
print('bb^2=',bb**2) #potencia
#
print(np.log(bb)) #logaritmo natural
print(np.log10(bb)) #logaritmo base 10
#
vr=np.random.randint(low=1, high=10, size=5)
print('vr=',vr)
#
#ordenar de mayor a menor
print('vr_s=',np.sort(vr))
#
print('minimo=',np.min(vr))
print('maximo=',np.max(vr))
#
# tamaño y forma del vector
print(np.size(a)) #numero de elementos
print(np.shape(b)) #dimensiones del arreglo

