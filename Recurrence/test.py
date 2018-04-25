import sys
#sys.path.insert(0, r'/local/scratch/sam5g13/library_folder')
#import library as lib
import matplotlib.pyplot as plt
import subprocess
from subprocess import Popen, PIPE
import numpy as np
from numpy import linalg as LA
import pyrqa
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
#from pyrqa.computation import RQAComputation
import os
import os.path

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

def water_delay(block_size):

	"""Calculates the correlations within the file, produces delay vectors based on this information and plots them graphically"""

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"
	file_name = "{}/tip4p2005_50_TOTEST.npy".format(directory)
	gnuplot = r'/usr/bin/gnuplot'


	file_data = np.load(file_name, mmap_mode='r')

	_, _, _, gamma, _ = file_data
	print len(gamma)

	a = blocksav(gamma, block_size)
	print len(a)	


	

water_delay(100)

























	

