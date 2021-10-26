import numpy as np
import matplotlib.pyplot as plt #libreria para graficos

#%matplotlib auto #muestra las gráficas en una miniventana nueva

dirvel='velocity.txt'#direccipon del archivo

datos=np.loadtxt(dirvel,dtype='float',delimiter=',',unpack='True',skiprows=1,
                 comments='#')[0:2,0:37] #Entre corchetes va el tamaño de los datos que queremos sacar [cols, filas]
col,fil=datos.shape #numero de columnas (col) y filas (fil)
print('numero de columnas y filas= ',datos.shape)
print(datos) #matriz que recibió los datos intercambiando filas y cols del archivo original

i=0  #inicializa la variable
while i in range(0,col-1) :
      t=datos[i,0:fil] #tiempo
      i+=1  #i=i+1
      vx=datos[i,0:fil] #componente vx
      i+=1
      # vy=datos[i,0:fil] #componete vy
      # i+=1
      # vz=datos[i,0:fil] #componente vz
      # i+=1

plt.figure(1)      
plt.plot(t,vx,'r')#,t,vy,'b',t,vz,'g')
plt.title('Graficando datos usando variables t, vx')

plt.figure(2)
plt.plot(datos[0,0:fil],datos[1,0:fil],'r')
plt.title('Graficando datos usando listas de datos')

#grafico 3d parametrico
fig = plt.figure(3)
ax = fig.gca(projection='3d')
ax.plot(vx,vx,0)
plt.title('Graficando datos en 3D')
plt.show() #muestra el grafico
     
