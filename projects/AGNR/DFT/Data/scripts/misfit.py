#!/usr/bin/python3

import numpy as np 
import scipy.integrate as integrate
import os
import matplotlib.pyplot as plt


files=os.popen("ls | sed -n '/^transmission/p'").read().splitlines()
N=len(files)


def f(a):
    inputTRC=open(a , 'r')
    inputemp=inputTRC.read().splitlines()
    
    for temp in range(len(inputemp)):
        inputemp[temp]=inputemp[temp].split()
      
    inputemp=np.array(inputemp).astype(float)
    return inputemp.transpose()

def g(a):
    
    inputTRC=open(a , 'r')
    inputemp=inputTRC.read().splitlines()
    l=len(inputemp)
    del(inputemp[l-1])
    
    for temp in range(len(inputemp)):
        inputemp[temp]=inputemp[temp].split()

    inputemp=np.array(inputemp)
    inputemp=np.delete(inputemp, 3, 1)
    inputemp=np.delete(inputemp, 3, 1)
    inputemp=np.delete(inputemp, 5, 1)
    inputemp=np.delete(inputemp, 4, 1)

    inputemp=inputemp.astype(float)
    return inputemp.transpose()

y1=g('transmission1.01/realization1.TRC')[[1], :]
data=[]
for temp in files:
    x=f(temp+'/average.TRC')[[0], :]
    y2=f(temp+'/average.TRC')[[1], :]
    y3=pow(y2-y1,2)
    I=integrate.simps(y3,x)
    data.append([float(temp[12:]), float(I)])
data=np.array(data)


x=data.transpose()[0]
y=data.transpose()[1]

#write misfit to file:
misfit=open('misfit-1.01', 'w')
misfit.writelines('%s  %s \n' %(x[l], y[l]) for l in range(len(x)) )

from scipy.interpolate import make_interp_spline, BSpline

x_smooth=np.linspace(x.min(),x.max(),300)

spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)

plt.axvline(x=0.14, ymin=0, ymax=1, ls ='--', color='black')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
#plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('$\chi$', fontsize=14)
plt.xlabel('n(%)', fontsize=14)
plt.plot(x_smooth, y_smooth, color='blue')
plt.plot(x, y, 'ro')
plt.show() 




