import sys
import matplotlib.pyplot as plt
import subprocess
from subprocess import Popen, PIPE
import numpy as np
from numpy import linalg as LA

import pyrqa
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

from pyrqa.computation import RecurrencePlotComputation
from pyrqa.image_generator import ImageGenerator

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

	gamma_sample = blocksav(gamma, block_size)

	gamma_file = "{}/tip4p2005_50_blocksize_{}_gamma.txt".format(directory, block_size)
	with open(gamma_file, 'w') as outfile:
		np.savetxt(outfile, gamma_sample)

	gamma_file_name = "{}/tip4p2005_50_blocksize_{}_gamma.txt".format(directory, block_size)

	correlations = subprocess.check_output(["corr", gamma_file_name])
	
	mutual_information = subprocess.check_output(["mutual", gamma_file_name])

	correlation_array = np.array(correlations.split()[5:], dtype=float)
	mutual_information_array = np.array(mutual_information.split()[2:], dtype=float)

	idx_odd = range(1,199,2)
	idx_even = range(0,200,2)

	idx_odd1 = range(1,43,2)
	idx_even1 = range(0,44,2)

	#correlation_values = correlation_array[idx_odd]
	mutual_information_values = mutual_information_array[idx_odd1]
	print 'LOOK HERE...........................................', mutual_information_array[idx_odd1], len(mutual_information_array[idx_odd1])

	"""
	delay_length = 0

	for o in range(len(correlation_values) - 1):
		print o, correlation_values[o], correlation_values[o+1]
		if correlation_values[o] > correlation_values[o+1]:
			delay_length = o 
		else: break
	
	delay_length = delay_length + 1

	print "The delay length is", delay_length
	"""

	mutual_info_length = 0

	for o in range(len(mutual_information_values) - 1):
		#print o, correlation_values[o], correlation_values[o+1]
		if mutual_information_values[o] > mutual_information_values[o+1]:
			mutual_info_length = o 
		else: break
	
	mutual_info_length = mutual_info_length + 1
	
	print "The mutual info length is", mutual_info_length

	#assert 	delay_length == mutual_info_length, "The minimums of the mutual information and the correlations are not equal! %d %d" % (delay_length, mutual_info_length)
	
	produce_delays = subprocess.check_output(["delay", gamma_file_name, "-d" + str(mutual_info_length)])

	
	delay_file = "{}/tip4p2005_50_blocksize_{}_gamma_delay_{}.txt".format(directory, block_size, mutual_info_length)
	f = open(delay_file, 'w')
	f.write(produce_delays)
	f.close()

	"""

	print produce_delays
	print len(produce_delays), len(mutual_information_values)
	plt.figure("produce_delays vs mutual information")
	plt.xlabel("produce_delays")
	plt.ylabel("Mutual information")
	plt.plot(produce_delays, mutual_information_values)
	plt.show()
	
	"""
	
	embedding = subprocess.check_output(["false_nearest", gamma_file_name])

	embedding_dimension = int(raw_input("What embedding dimension would you like to use? "))
	
	run_calc = subprocess.check_output(['gnuplot', '-e', "filename='{}/tip4p2005_50_blocksize_{}_gamma_delay_{}.txt';ofilename='tip4p2005_50_blocksize_{}_gamma_delay_{}_graph.png'".format(directory, block_size, mutual_info_length, block_size, mutual_info_length ),"plot.gnu"])


	"""Imports the time series and specifies each aspect used in building the recurrence matrix"""

	settings = Settings(time_series = gamma_sample, embedding_dimension = embedding_dimension, time_delay = mutual_info_length, similarity_measure = EuclideanMetric, neighbourhood = FixedRadius(radius = 13), min_diagonal_line_length = 2, min_vertical_line_length = 2)

	"""Performs the computation and prints out all the results"""

	rqacomputation = RQAComputation.create(settings, verbose = True)

	rqaresult = rqacomputation.run()

	print rqaresult

	"""Creates the Recurrence matrix for viewing"""

	rpcomputation = RecurrencePlotComputation.create(settings)

	rpresult = rpcomputation.run()

	ImageGenerator.save_recurrence_plot(rpresult.recurrence_matrix, 'recurrence_plot.png')


"""
x = [10, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 200000, 400000]
for i in range(len(x)):
	water_delay(x[i]) 	
"""
water_delay(100)





