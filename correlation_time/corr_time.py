import sys
sys.path.insert(0, r'/local/scratch/sam5g13/library_folder')
#import library as lib
import numpy as np
import matplotlib.pyplot as plt


#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#""""" Taking the equation for correlation time from Frenkel and Smit"""""
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""BLOCK AVERAGING"""

def blockslice(A, tb):
	nb = len(A)/tb
	blocks = np.zeros((nb, tb))
	for i in range(nb):
		blocks[i] += A[i*tb: (i+1)*tb] 
	return blocks

def blocksav(A, tb):
	nb = len(A)/tb
	blocks = np.zeros(nb)
	for i in range(nb):
		blocks[i] += np.mean(A[i*tb: (i+1)*tb])
	return blocks

"""Average correlation of A"""

def Ca(tau, A, dt):
	tau_0 = len(A)
	
	Sum = 0	
	
	for t in range(tau_0 - tau):
		Sum += A[t]*A[t+tau]
	Sum = Sum*(1./tau_0)*dt # x 1 time step

	return Sum

"""Correlation of A"""

def autocorr(x, dt):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:] * dt / len(x)

N_files = 1
TotSurf = []

for i in range(N_files):
	print 'This is file', i
	E = open('/local/scratch/sam5g13/AMBER/ARGON/T_85_K/CUT_10_A/A_TEST/A_20/SURFACE_2/argon_50_surface{}.{}'.format(i, 'out'), 'r')
	
	lines = E.readlines()
	E.close()
	length = len(lines)

	surften = []

	for line in lines:
        	templine = line.split()
		if len(templine) != 0 :
			if templine[0] == 'SURFTEN': 
				surften.append(float(templine[2]))
			if templine[0] == 'A' and templine[1] == 'V' :break

	TotSurf = TotSurf + surften

print len(TotSurf)
ca = []

for tau in range(1,len(TotSurf)):
	ca.append(Ca(tau, TotSurf, 1))

numpy_ca = autocorr(TotSurf, 1)


plt.figure(0)
plt.plot(range(len(ca)), ca, color = 'green')

plt.figure(1)
plt.plot(range(len(numpy_ca)),numpy_ca)


plt.show()
