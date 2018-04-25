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

def blocksav(A, tb):
	nb = len(A)/tb
	blocks = np.zeros(nb)
	for i in range(nb):
		blocks[i] += np.mean(A[i*tb: (i+1)*tb])
	return blocks

def block_error(A, ntb):
	
	prod = np.zeros(ntb)
	for tb in range(1,ntb):
		stdev1 = np.var(blocksav(A, tb))
		prod[tb] += stdev1

	var2 = np.var(A)

	pt = [(tb+1)*prod[tb]/var2 for tb in range(ntb)]

	block_error_x = [(1./tb) for tb in range(2,ntb+1)]
	block_error_y = [(1./pt[i]) for i in range(1,ntb)]
	
	_, intercept, _, _, _ = stats.linregress(block_error_x,block_error_x)

	M = len(A)
	corr_err = (np.std(A) * np.sqrt(1. / (intercept*M)))

	plt.figure('BLOCK ERROR')
	plt.plot(block_error_x, block_error_x)
	

	plt.figure('CORRELATION TIME')
	plt.plot(range(1,ntb+1), pt)
	plt.show()

	return pt, corr_err


def ARGON_plot_correlation_info_v_temp(directory, block_size):

	"""

	Plots the correlation time against the temperature for Argon

	Parameters 
	----------

	directory : string
		The file path for where the file is stored

	"""

	#directory = '/local/scratch/sam5g13/AMBER/ARGON'

	temp_range = np.arange(85, 146, 5) # The temperatures that each simulation is run at
	cutoff_range = [22] # The cutoff range the simulation is run at, no correction terms are implemented
	array_correlation_info_store = [] # Array to store the correlation information of each temperature

	for temp in temp_range:
	
		"""For loop ranges over each different temperature file"""

		file_name = "{}/surface_ten_data/argon_{}_22_6000_ST.npy".format(directory, temp)

		file_data = np.load(file_name, mmap_mode='r')

		gamma = file_data
		print 'HERE', len(gamma)

		gamma_sample_correlation = blocksav(gamma, block_size)

		gamma_file = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)
		with open(gamma_file, 'w') as outfile:
			np.savetxt(outfile, gamma_sample_correlation)

		gamma_file_name = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)

		"""Correlation Information"""

		correlation = subprocess.check_output(["corr", gamma_file_name]) 
		
		tempfile = open('corr.dat', 'w')
		correlation_info_array = np.array(correlation.split()[5:], dtype=float)
		tempfile.write(correlation_info_array)
		tempfile.close()

		idx_odd = range(1, len(correlation_info_array) + 1, 2)
		idx_even = range(0, len(correlation_info_array), 2)

		correlation_delay = correlation_info_array[idx_even]
		correlation_values = correlation_info_array[idx_odd]

		correlation_info_x = correlation_delay
		correlation_info_y = correlation_values
	
		plt.figure('CORRELATION INFORMATION')
		plt.xlabel('Time delay')
		plt.ylabel('CORRELATION information')
		plt.scatter(correlation_info_x, correlation_info_y)
	
		plt.figure('CORRELATION INFORMATION')
		plt.plot(correlation_info_x, correlation_info_y)
	

		minimum_counter = 0
	
		for i in range(len(correlation_values) - 1):

			if correlation_values[i] > correlation_values[i+1]: 
				minimum_counter = i

			else: break


		first_min_correlation_info_ARGON = minimum_counter + 1
		print first_min_correlation_info_ARGON, 'PRINTED HERE'

		array_correlation_info_store.append(first_min_correlation_info_ARGON)

	print len(temp_range), len(array_correlation_info_store)
	plt.figure('TEMP V CORRELATION INFORMATION')
	plt.xlabel('Temperature')
	plt.ylabel('1st minimum of the correlation information - delay value')
	plt.plot(temp_range, array_correlation_info_store)
	plt.show()

	


def ARGON_plot_mutual_info_v_temp(directory, block_size):

	"""

	Plots the mutual information against the temperature for Argon

	Parameters 
	----------

	directory : string
		The file path for where the file is stored

	"""

	#directory = '/local/scratch/sam5g13/AMBER/ARGON'

	temp_range = np.arange(85, 146, 5) # The temperatures that each simulation is run at
	cutoff_range = [22] # The cutoff range the simulation is run at, no correction terms are implemented
	array_mutual_info_store = [] # Array to store the mutual information of each temperature

	for temp in temp_range:
	
		"""For loop ranges over each different temperature file"""

		file_name = "{}/surface_ten_data/argon_{}_22_6000_ST.npy".format(directory, temp)

		file_data = np.load(file_name, mmap_mode='r')

		gamma = file_data
		print 'HERE', len(gamma)

		gamma_sample_mutual = blocksav(gamma, block_size)

		gamma_file = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)
		with open(gamma_file, 'w') as outfile:
			np.savetxt(outfile, gamma_sample_mutual)

		gamma_file_name = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)

		"""Mutal Information"""

		mutual = subprocess.check_output(["mutual", gamma_file_name]) 
		
		tempfile = open('mutual.dat', 'w')
		mutual_info_array = np.array(mutual.split()[2:], dtype=float)
		tempfile.write(mutual_info_array)
		tempfile.close()

		idx_odd = range(1, len(mutual_info_array) + 1, 2)
		idx_even = range(0, len(mutual_info_array), 2)

		mutual_delay = mutual_info_array[idx_even]
		mutual_values = mutual_info_array[idx_odd]

		mutual_info_x = mutual_delay
		mutual_info_y = mutual_values
	
		plt.figure('MUTUAL INFORMATION')
		plt.xlabel('Time delay')
		plt.ylabel('Mutual information')
		plt.scatter(mutual_info_x, mutual_info_y)
	
		plt.figure('MUTUAL INFORMATION')
		plt.plot(mutual_info_x, mutual_info_y)
	

		minimum_counter = 0
	
		for i in range(len(mutual_values) - 1):

			if mutual_values[i] > mutual_values[i+1]: 
				minimum_counter = i

			else: break


		first_min_mutual_info_ARGON = minimum_counter + 1
		print first_min_mutual_info_ARGON, 'PRINTED HERE'

		array_mutual_info_store.append(first_min_mutual_info_ARGON)

	print len(temp_range), len(array_mutual_info_store)
	plt.figure('TEMP V MUTUAL INFORMATION')
	plt.xlabel('Temperature')
	plt.ylabel('1st minimum of the mutual information - delay value')
	plt.plot(temp_range, array_mutual_info_store)
	plt.show()


def produce_rp_ARGON(block_size, num_delays):

	"""
	directory = "/local/scratch/sam5g13/AMBER/ARGON/T_85_K/CUT_10_A/A_TEST/A_20/SURFACE_2/DATA/THERMO"
	file_name = "{}/argon_4000_TOT_E_ST.npy".format(directory)
	gnuplot = r'/usr/bin/gnuplot'
	"""

	directory = "/local/scratch/sam5g13/AMBER/ARGON/surface_ten_data"

	temp = 85

	#time_series = FileReader.file_as_float_array('/local/scratch/sam5g13/AMBER/ARGON/surface_ten_data/argon_{}_22_6000_ST.npy'.format(temp), column = 0)

	file_name = "{}/argon_{}_22_6000_ST.npy".format(directory, temp)

	file_data = np.load(file_name, mmap_mode='r')

	gamma = file_data
	print 'HERE', len(gamma)

	gamma_sample = blocksav(gamma, block_size)

	gamma_file = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)
	with open(gamma_file, 'w') as outfile:
		np.savetxt(outfile, gamma_sample)

	gamma_file_name = "{}/ARGON_blocksize_{}_gamma_temp_{}.txt".format(directory, block_size, temp)
	print len(gamma_sample), len(gamma_file_name), 'LOOK HERE', len(file_data), gamma_file_name
	sys.exit()
          
	"""Mutal Information"""

	print type(gamma_file_name)

	mutual = subprocess.check_output(["mutual", gamma_file_name]) 
	
	tempfile = open('mutual.dat', 'w')
	mutual_info_array = np.array(mutual.split()[2:], dtype=float)
	tempfile.write(mutual_info_array)
	tempfile.close()

	idx_odd = range(1, len(mutual_info_array) + 1, 2)
	idx_even = range(0, len(mutual_info_array), 2)

	mutual_delay = mutual_info_array[idx_even]
	mutual_values = mutual_info_array[idx_odd]

	mutual_info_x = mutual_delay
	mutual_info_y = mutual_values

	plt.figure('MUTUAL INFORMATION')
	plt.scatter(mutual_info_x, mutual_info_y)

	plt.figure('MUTUAL INFORMATION')
	plt.plot(mutual_info_x, mutual_info_y)
	plt.show()

	minimum_counter = 0

	for i in range(len(mutual_values) - 1):

		if mutual_values[i] > mutual_values[i+1]: 
			minimum_counter = i

		else: break


	first_min_mutual_info_ARGON = minimum_counter + 1

	print 'The first minimum for the mutual information for the ARGON data  is', first_min_mutual_info_ARGON
	
	#"""Correlation Information"""
	"""
	corr = subprocess.check_output(["corr", gamma_file_name]) 
	
	tempfile = open('corr.dat', 'w')
	correlation_info_array = np.array(corr.split()[5:], dtype=float)
	tempfile.write(correlation_info_array)
	tempfile.close()

	idx_odd1 = range(1, len(correlation_info_array) + 1, 2)
	idx_even1 = range(0, len(correlation_info_array), 2)

	correlation_delay = correlation_info_array[idx_even1]
	correlation_values = correlation_info_array[idx_odd1]

	corr_x = correlation_delay
	corr_y = correlation_values

	plt.figure('CORRELATION INFORMATION')
	plt.scatter(corr_x, corr_y)

	plt.figure('CORRELATION INFORMATION')
	plt.plot(corr_x, corr_y)
	plt.show()

	minimum_counter1 = 0

	for k in range(len(correlation_values) - 1):

		if correlation_values[k] > correlation_values[k+1]: 
			minimum_counter1 = k

		else: break


	first_min_correlation_info_ARGON = minimum_counter1 + 1

	print 'The first minimum for the correlation information for the ARGON data  is', first_min_correlation_info_ARGON
	"""
	"""Correlation time"""

	ntb = len(gamma)/block_size
	print len(gamma), block_size, ntb
	#block_error(gamma, ntb)
	"""
	"""

	"""Plotting the delays"""

	"""Reads in directory and file names and the data set"""

	"""Optimal embedding depends on the application"""

	"""delay and embed routines are an important tool for the visulisation inspection of data"""
	"""
	for j in range(num_delays):

		tempfile = open('delay.dat', 'w')
		delay = subprocess.check_output(["delay", gamma_file_name, "-d" + str(j + 1)]) 
		tempfile.write(delay)
		tempfile.close()

		delays_x = []
		delays_y = []

		E = open('delay.dat', 'r')
		lines = E.readlines()
		E.close()

		for line in lines:
			templine = line.split()
			delays_x.append(templine[0])
			delays_y.append(templine[1])
		plt.figure(j)
		plt.scatter(delays_x, delays_y)
	plt.show()
	"""
	"""False nearest neighbour search"""

	false_nearest = subprocess.check_output(["false_nearest", gamma_file_name, "-m1", "-M1,15", "-d" + str(first_min_mutual_info_ARGON)]) 

	split_file_name = np.array(false_nearest.split(), dtype = float)
	fnn_embedding= split_file_name[0::4]
	fnn_fraction = split_file_name[1::4]
	plt.figure('fnn vs embedding for ARGON data')
	plt.axis([1, 30, 0, 1.5])
	plt.plot(fnn_embedding, fnn_fraction)
	plt.show()

	#Imports the time series and specifies each aspect used in building the recurrence matrix

	#embedding_dimension = raw_input('What is the embedding dimension?: ')

	settings = Settings(time_series = gamma_sample, embedding_dimension = 14, time_delay = first_min_mutual_info_ARGON, similarity_measure = EuclideanMetric, neighbourhood = FixedRadius(radius = 25), min_diagonal_line_length = 2, min_vertical_line_length = 2)

	#Performs the computation and prints out all the results

	rqacomputation = RQAComputation.create(settings, verbose = True)

	rqaresult = rqacomputation.run()

	print rqaresult

	#Creates the Recurrence matrix for viewing

	rpcomputation = RecurrencePlotComputation.create(settings)

	rpresult = rpcomputation.run()

	ImageGenerator.save_recurrence_plot(rpresult.recurrence_matrix, 'recurrence_plot_temp_{}_block_size_{}.png'.format(temp, block_size))

		
directory = '/local/scratch/sam5g13/AMBER/ARGON'
#ARGON_plot_mutual_info_v_temp(directory, 1)
#ARGON_plot_correlation_info_v_temp(directory, 1)
#ARGON_plot_correlations_v_temp(directory)

produce_rp_ARGON(25, 10)

plt.show()
