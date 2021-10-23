# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:19:13 2021

@author: samue
"""

import matplotlib.pyplot as plt
import numpy as np

filename= 'Table_op.txt'

T,P=np.loadtxt(filename,delimiter='\t', usecols=(0,1), unpack=True)
#print('P',P)
plt.plot(T,P,'*')
plt.show()
#print(T)