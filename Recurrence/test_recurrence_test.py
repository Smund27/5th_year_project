import sys
sys.path.insert(0, r'/local/scratch/sam5g13/Sam_5th-yr_Project')
import matplotlib.pyplot as plt
import subprocess, os, sys
from subprocess import Popen, PIPE
import numpy as np
from numpy import linalg as LA
from tempfile import TemporaryFile
import csv
from pylab import *
#from scipy import stats 
import recurrence_library as lib

import pyrqa
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

from pyrqa.computation import RecurrencePlotComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.file_reader import FileReader


#length of data is 600,000
temp = 85
block_size = 299999
num_delays = 2

directory = "/local/scratch/sam5g13/AMBER/ARGON/surface_ten_data"
root = "{}/argon_{}_22_6000_ST.npy".format(directory, temp)

load_data = np.load(root, mmap_mode='r')

file_name_npy = lib.blocksav(load_data, block_size)

file_name_txt = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)
with open(file_name_txt, 'w') as outfile:
	np.savetxt(outfile, file_name_npy)

print len(file_name_txt), file_name_txt
"""
correlation = lib.TISEAN_correlation(file_name_txt)
plot_correlations = lib.TISEAN_plot_correlations(file_name_txt)
correlations_firstmin = lib.TISEAN_correlations_firstmin(file_name_txt)
"""

mutual_info = lib.TISEAN_mutualinfo(file_name_txt)
plot_mutual_info = lib.TISEAN_plot_mutualinfo(file_name_txt)
mutual_info_firstmin = lib.TISEAN_mutualinfo_firstmin(file_name_txt)

#delays = lib.TISIAN_plot_delays(file_name_txt, 15)

#false_nearest_neighbours = lib.TISEAN_fnn(file_name)
#RP = lib.recurrence_plot_maker(file_name_txt, embedding, time_delay, neighbourhood)








"""	
blocks = np.zeros(y)
for i in range(y):
	blocks[i] += np.mean(test_data[i*x: (i+1)*x])
	#print blocks[i],',', i,',', x,',', test_data[i*x],',',  i*x,',',  (i+1)*x,',',  test_data[i*x: (i+1)*x]



print blocks
def Sam_block_error(file_name, num_blocks):

	array_of_block_averages = []
	
	num_steps_per_block = len(file_name)/num_blocks

	for i in range(num_blocks):
	
		array_of_block_averages.append(np.mean(file_name[i * num_steps_per_block: (i + 1) * num_steps_per_block]))

	block_average_variance	= np.var(array_of_block_averages)

	statistical_inefficiency = []

	print range(num_steps_per_block)

	for j in range(1, num_blocks):
	
		num_steps_per_block = len(file_name)/num_blocks	

		statistical_inefficiency.append(np.float((j * block_average_variance) / np.var(file_name)))

	plt.scatter(range(num_steps_per_block), statistical_inefficiency)
	plt.show()

	return array_of_block_averages, num_steps_per_block, block_average_variance
	
Sam_block_error(load_data, 100)

	
"""		

	
