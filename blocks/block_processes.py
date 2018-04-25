import sys
import matplotlib.pyplot as plt
import subprocess, os, sys
from subprocess import Popen, PIPE
import numpy as np
from numpy import linalg as LA
from tempfile import TemporaryFile
import csv
from pylab import *
from scipy import stats 

import pyrqa
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

from pyrqa.computation import RecurrencePlotComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.file_reader import FileReader

def block_error(A, ntb):

	prod = np.zeros(ntb)
	for tb in range(1,ntb):
		stdev1 = np.var(blocksav(A, tb))
		prod[tb] += stdev1

	var2 = np.var(A)

	pt = [(tb+1)*prod[tb]/var2 for tb in range(ntb)]

	x = [(1./tb) for tb in range(2,ntb+1)]
	y = [(1./pt[i]) for i in range(1,ntb)]
	
	_, intercept, _, _, _ = stats.linregress(x,y)

	M = len(A)
	corr_err = (np.std(A) * np.sqrt(1. / (intercept*M)))

	return pt, corr_err

def blocksav(A, tb):
	nb = len(A)/tb
	blocks = np.zeros(nb)
	for i in range(nb):
		blocks[i] += np.mean(A[i*tb: (i+1)*tb])
	return blocks

def autocorr(x):

	result = np.correlate(x, x, mode='full')

	return result[result.size/2:]

directory = "/local/scratch/sam5g13/AMBER/ARGON"
block_directory = '/local/scratch/sam5g13/AMBER/ARGON'

temp_range = np.arange(85, 146, 5) # The temperatures that each simulation is run at 

block_size = np.arange(5,51, 5)

#for temp in temp_range:

temp = 145

file_name = "{}/surface_ten_data/argon_{}_22_6000_ST.npy".format(directory, temp)

file_data = np.load(file_name, mmap_mode='r')

gamma = file_data

#Correlation times of each temperature
correlation_time_array = [914.14993663153678, 836.09009332764981, 784.70347539227976, 649.0238688715117, 589.10676090153959, 500.25251504722712, 473.35258783693621, 439.68240587226683, 423.45256310211704, 348.3362585828271, 329.76761866587134, 318.82680244383954, 295.97080602371614]

#Number of blocks is 600000 divided 
tb = int(600000/914.14)
print tb

x = blocksav(gamma, tb)

#print len(x), x

plt.plot(range(len(x)), x)
plt.show()






























