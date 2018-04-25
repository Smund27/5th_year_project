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
	print nb, len(A), tb
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

def TISEAN_correlation(file_name):

	correlation = subprocess.check_output(["corr", "-D1000", file_name])

	return correlation

def TISEAN_plot_correlations(file_name):

	correlation = TISEAN_correlation(file_name)

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
	plt.xlabel('TIME DELAY')
	plt.ylabel('CORRELATION INFORMATION')
	plt.scatter(correlation_info_x, correlation_info_y)

	plt.figure('CORRELATION INFORMATION')
	plt.plot(correlation_info_x, correlation_info_y)

	plt.show()

def TISEAN_correlations_firstmin(file_name):

	correlation = subprocess.check_output(["corr", file_name])

	tempfile = open('corr.dat', 'w')
	correlation_info_array = np.array(correlation.split()[5:], dtype=float)
	tempfile.write(correlation_info_array)
	tempfile.close()

	idx_odd = range(1, len(correlation_info_array) + 1, 2)
	idx_even = range(0, len(correlation_info_array), 2)

	correlation_delay = correlation_info_array[idx_even]
	correlation_values = correlation_info_array[idx_odd]	
	
	minimum_counter = 0

	for i in range(len(correlation_values) - 1):

		if correlation_values[i] > correlation_values[i+1]: 
			minimum_counter = i

		else: break


	first_min_correlation_info = minimum_counter + 1

	return first_min_correlation_info


def TISEAN_mutualinfo(file_name):

	mutual = subprocess.check_output(["mutual", file_name]) 
	
	return mutual

def TISEAN_plot_mutualinfo(file_name):
	
	mutual = subprocess.check_output(["mutual", file_name]) 

	
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

	plt.show()

def TISEAN_mutualinfo_firstmin(file_name):

	mutual = subprocess.check_output(["mutual", file_name]) 

	tempfile = open('mutual.dat', 'w')
	mutual_info_array = np.array(mutual.split()[2:], dtype=float)
	tempfile.write(mutual_info_array)
	tempfile.close()

	idx_odd = range(1, len(mutual_info_array) + 1, 2)
	idx_even = range(0, len(mutual_info_array), 2)

	mutual_delay = mutual_info_array[idx_even]
	mutual_values = mutual_info_array[idx_odd]

	minimum_counter = 0

	for i in range(len(mutual_values) - 1):

		if mutual_values[i] > mutual_values[i+1]: 
			minimum_counter = i

		else: break


	first_min_mutual_info = minimum_counter + 1

	return first_min_mutual_info

def TISIAN_plot_delays(file_name, num_delays):

	for j in range(num_delays):

		tempfile = open('delay.dat', 'w')
		delay = subprocess.check_output(["delay", file_name, "-d" + str(j + 1)]) 
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

def TISEAN_fnn(file_name):

	first_min_mutual_info = TISEAN_mutualinfo_firstmin(file_name)

	false_nearest = subprocess.check_output(["false_nearest", file_name, "-m1", "-M1,25", "-d" + str(first_min_mutual_info)]) 

	split_file_name = np.array(false_nearest.split(), dtype = float)
	fnn_embedding= split_file_name[0::4]
	fnn_fraction = split_file_name[1::4]
	plt.figure('fnn vs embedding for data')
	plt.axis([1, 30, 0, 1.5])
	plt.plot(fnn_embedding, fnn_fraction)
	plt.show()

def recurrence_plot_maker(file_name, embedding, time_delay, neighbourhood):

	file_name = blocksav(gamma, block_size)

	settings = Settings(time_series = file_name, embedding_dimension = embedding, time_delay = time_delay, similarity_measure = EuclideanMetric, neighbourhood = FixedRadius(radius = neighbourhood), min_diagonal_line_length = 2, min_vertical_line_length = 2)

	#Performs the computation and prints out all the results

	rqacomputation = RQAComputation.create(settings, verbose = True)

	rqaresult = rqacomputation.run()

	print rqaresult

	#Creates the Recurrence matrix for viewing

	rpcomputation = RecurrencePlotComputation.create(settings)

	rpresult = rpcomputation.run()

	ImageGenerator.save_recurrence_plot(rpresult.recurrence_matrix, 'recurrence_plot_temp_{}_block_size_{}.png'.format(temp, block_size))






 
