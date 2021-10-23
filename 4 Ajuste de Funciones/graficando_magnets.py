"""
Created on Fri Apr 16 18:01:44 2021

@author: JuanGuarin y Juan Diego Figueroa
"""

## Importación de librerias
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

## Importación de datos
"""Importar con xlrd
   Primero se ebe poner la direccion del archivo de excel, poniendo barras inclinadas dobles: \\
   y así, evitar errores en la ejecución. En la dirección debe ir el nombre del archivo con la
   extension .xlsx que corresponde a todo archivo de excel.
"""
#dirección en el computador del excel
direccion="Averages.xlsx" 
book = xlrd.open_workbook(direccion) #Se importan todos los datos en la variable book
sheet = book.sheet_by_name("nsuperficie") #se importa la hoja llaada nsuperficie

print('el valor de la casilla (4,2) es:',sheet.cell_value(3,1)) #se imprime el valor de la casilla en la fila 4 columna 2 del archivo excel
rows, cols=sheet.nrows, sheet.ncols #filas y columnas de sheet


#Se crean los vectores donde almacenaremos los datos de tamaño rows-1, ya que, la primer fila del excel
#contiene las etiquetas de lo que representa cada columna

t=np.zeros(rows-1)  #se crea el vector que almacenara los datos de tiempo
y=np.zeros(rows-1)  #se crea el vector que almacenara los datos de posición en Y
dy=np.zeros(rows-1) #se crea el vector que almacenara los datos de errores de la posición en Y
 

## Recoleeción de datos
for i in range(rows-1): #recorre filas
    a=sheet.cell_value(i+1, 0) #este dato corresponden a la vaiable tiempo
    b=sheet.cell_value(i+1, 1) #esto dato corresponden a la vaiable posición en Y
    c=sheet.cell_value(i+1, 4) #esto dato corresponden a la vaiable errores de la posición en Y
    
    if type(a) == float: #si el valor de a es un real se guarda en el vector de tiempo
        t[i]=a  
    
    if type(b) == float: #si el valor de b es un real se guarda en el vector de posición en Y
        y[i]=b
    
    if type(c) == float: #si el valor de c es un real se guarda en el vector de errores de la posición en Y
        dy[i]=c
           
        
""""
Defina la función que se quiera ajustar. La entrada debe ser la o las variables independientes (x)
y los parametros de ajuste, en este caso los valores a,b,c
"""
def funcion(x, a, b,c):  
    return a*x**2 + b*x + c


""""
Ajusta la función de un conjunto de datos tomando en cuenta los errores. Las entradas son: la funcion para 
ajustar, los datos de la variable independiente, los datos de la variable dependiente, o errores de los datos
y absolute_sigma = True que toma en cunta errores ponderads.
"""
t_error = 1/240 #incertidumbre en el tiempo
y_error = dy #incertidumbre en la posición

popt1, pcov = curve_fit(funcion, t, y, sigma=dy, absolute_sigma=True)
print('vector con los paraametros de la funcion de ajuste: ',popt1)                     #Parametros de la funcion ajustada, en este caso son los valores de a, b y c
                                 #cada uno de estos datos estan almacenados en popt1 como un vector de tamaño igual al numero de parametros
                                 #se almacenan en orden
                                 
print('vector con las desviaciones estandar de los parametros de la funcion de ajuste: ',np.sqrt(np.diag(pcov)))    #Diagonal de covarianza que son las desviaciones estandar de los parametros a,b,c
print('a=',popt1[0]) #Imprime el valor calculado en la funcion de ajuste para el parametro a
print('b=',popt1[1]) #Imprime el valor calculado en la funcion de ajuste para el parametro b
print('c=',popt1[2]) #Imprime el valor calculado en la funcion de ajuste para el parametro c

## Graficando datos
fig = plt.figure(figsize=(10, 5)) #Se crea la ventana donde se graficaran los datos, con tamaño de 10x5
#Los caps ponen las barras de error en cada uno de los puntos de datos
(_, caps, _) = plt.errorbar(t, y, xerr= t_error, yerr= dy, fmt='ob', markersize=5, capsize=4, ecolor = 'blue', label='$Magnets$')
#se cambia el tamaño de las barrar de error segun se desee, el valor de 1 indica que todas las barras de error se multiplican por 1, es decir, quedan igual
for cap in caps:
    cap.set_markeredgewidth(1) 

#Grafica la funcion de ajuste como una linea segmentada de color rojo
plt.plot(t,funcion(t, popt1[0], popt1[1], popt1[2]),color = 'red', linestyle = '-', label=r'$y = %.3ft^2+%.3ft+%.5f$'%(popt1[0], popt1[1], popt1[2]))   

#Se hace una anotación en un punto especifico de la grafica. Se puede ajustar la posición con la variable xy.
plt.annotate(r'$R^2 = 0.99270$', xy=(0.02, 0.2),  xycoords='data', xytext=(+5, +5), textcoords='offset points', 
             color= 'black',fontsize=15, rotation = 0)

#Se definen titulos, tamaños de letras, grozor de los ticks. Los $$ permiten escribir información en un estilo de esirtura matemática 
#(ecuaciones, letras griegas, super y subindices, fracciones, etc)
plt.ylabel('$Position Y$ [$m$]', fontsize = 15) #etiqueta del eje y
plt.xlabel('$Time $ [$s$]', fontsize = 15) #etiqueta del eje x
plt.title('Metallic surface', fontsize = 18) #titulo del plot
plt.legend(fontsize = 15, loc=2) #Pone la leyenda, que le da nombre a cada grafica que se pone en pantalla
plt.xticks(size = 15) 
plt.yticks(size = 15)
plt.grid(True) #Pone una regilla o cuadricula de fondo
plt.show() #muestra todo en la grafica
print('correlación de los datos: ', np.corrcoef(t,y))         #Imprime la correlación de los datos (cuando se trata de un ajuste de linea recta)