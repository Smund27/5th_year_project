import sys
import matplotlib.pyplot as plt
import subprocess
from subprocess import Popen, PIPE
import numpy as np
from numpy import linalg as LA
from tempfile import TemporaryFile
import csv

import pyrqa
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

from pyrqa.computation import RecurrencePlotComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.file_reader import FileReader


def create_sine_test_data():

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"

	time = np.arange(0, 100, 0.1)
	
	amplitude = np.sin(time)

	plt.plot(time, amplitude)
	plt.show()

	sine_file = "{}/test_sine_data_AMPLITUDE.txt".format(directory)
	with open(sine_file, 'w') as outfile:
		np.savetxt(outfile, amplitude)
	"""
	#sine = 

	Fs = 8000
	frequency = 5
	sample = 8000
	x = np.arange(sample)
	y = np.sin(2 * np.pi * frequency* x / Fs)
	plt.plot(x, y)
	plt.show()

	z = np.sqrt(x**2 + y**2)
	print len(x), len(y), len(z)
	plt.plot(range(len(z)), z)
	plt.show()

	sys.exit()

	sine_file = "{}/test_sine_data_frequency_{}.txt".format(directory, frequency)
	with open(sine_file, 'w') as outfile:
		np.savetxt(outfile, (x,y))
	"""
	

def sine_data_rp_test():

	a = []

	for i in range(1000):

		a.append(i * 2 * np.pi/67.0)
		
	X = np.sin(a)

	return X	


def HenonMap(a, b, x, y):
	
	return y + 1.0 - a * x * x, b * x


def TISEAN_henon_data():

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"

	henon_data = subprocess.check_output(["henon", "-l1000"])
	hennon_data_array = henon_data.split(" ")
	hennon_data_array[:] = [item for item in hennon_data_array if item != '' and item !='\n']
	henon_data_array = np.array(hennon_data_array[:], dtype = float)

	
	idx_odd = range(1, len(henon_data_array) + 1, 2)
	idx_even = range(0, len(henon_data_array), 2)

	x = []
	y = []
	
	for i in idx_even:
		xtemp = henon_data_array[i]
		x.append(xtemp)
	
	for j in idx_odd:
		ytemp = henon_data_array[j]
		y.append(ytemp)
	
	henon_file = "{}/test_henon_data.txt".format(directory)
	with open(henon_file, 'w') as outfile:
		np.savetxt(outfile, (x,y))

	henon_file_x_coord = "{}/test_henon_x_coord_data.txt".format(directory)
	with open(henon_file_x_coord, 'w') as outfile:
		np.savetxt(outfile, x)

	henon_file_y_coord = "{}/test_henon_y_coord_data.txt".format(directory)
	with open(henon_file_y_coord, 'w') as outfile:
		np.savetxt(outfile, y)

	henon_file_r_coord = "{}/test_henon_r_coord_data.txt".format(directory)
	with open(henon_file_r_coord, 'w') as outfile:
		np.savetxt(outfile, np.sqrt(np.array(x)**2 + np.array(y)**2))
	

def find_correlation_time(data_set, resolution):

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"

	if data_set == 'sine':

		"""Reads in directory and file names and the data set"""

		file_name = "{}/test_sine_data_AMPLITUDE.txt".format(directory)

		correlations = subprocess.check_output(["corr", file_name]) 
		#print 'CORRELATIONS HERE', correlations
		
		tempfile = open('corr.dat', 'w')
		correlation_array = np.array(correlations.split()[5:], dtype = float)
		tempfile.write(correlation_array)
		tempfile.close()

		idx_odd = range(1, len(correlation_array) + 1, 2)
		idx_even = range(0,len(correlation_array), 2)

		correlation_index = correlation_array[idx_even]
		correlation_values = correlation_array[idx_odd]

		x = correlation_index
		y = correlation_values
		
		plt.figure('correlation_index vs correlation_values')
		plt.scatter(x, y)
		
		plt.plot(x, y)
		plt.show()
		
		minimum_counter = 0

		for i in range(len(correlation_values) - 1):

			if correlation_values[i] > 0: 
				minimum_counter = i

			else: break
	

		first_min_autocorr_sine = minimum_counter

		print 'The first minimum for the autocorrelation function for the sine wave is', first_min_autocorr_sine

		return first_min_autocorr_sine

	if data_set == 'amplitude':
		
		"""Reads in directory and file names and the data set"""

		file_name = "{}/amplitude.dat".format(directory)

		correlations = subprocess.check_output(["corr", file_name]) 
		
		tempfile = open('corr.dat', 'w')
		correlation_array = np.array(correlations.split()[5:], dtype=float)
		tempfile.write(correlation_array)
		tempfile.close()

		idx_odd = range(1, len(correlation_array) + 1, 2)
		idx_even = range(0,len(correlation_array), 2)

		correlation_index = correlation_array[idx_even]
		correlation_values = correlation_array[idx_odd]

		x = correlation_index
		y = correlation_values
		
		plt.figure(0)
		plt.scatter(x, y)
		
		plt.figure(1)
		plt.plot(x, y)
		plt.show()
		
		minimum_counter = 0

		for i in range(len(correlation_values)):

			if correlation_values[i] > 0: 
				minimum_counter = i

			else: break
	

		first_min_autocorr_amplitude = minimum_counter 

		print 'The first minimum for the autocorrelation function for the amplitude data is', first_min_autocorr_amplitude

		return first_min_autocorr_amplitude

def find_mutual_information(data_set, resolution):

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"

	if data_set == 'sine':

		"""Reads in directory and file names and the data set"""

		file_name = "{}/test_sine_data_AMPLITUDE.txt".format(directory)

		mutual = subprocess.check_output(["mutual", file_name]) 
		
		tempfile = open('mutual.dat', 'w')
		mutual_info_array = np.array(mutual.split()[2:], dtype=float)
		tempfile.write(mutual_info_array)
		tempfile.close()

		idx_odd = range(1, len(mutual_info_array) + 1, 2)
		idx_even = range(0,len(mutual_info_array), 2)

		mutual_delay = mutual_info_array[idx_even]
		mutual_values = mutual_info_array[idx_odd]

		x = mutual_delay
		y = mutual_values
		
		plt.figure('MUTUAL INFO SINE')
		plt.scatter(x, y)
		
		plt.figure('MUTUAL INFO SINE')
		plt.plot(x, y)
		plt.show()
		
		minimum_counter = 0

		for i in range(len(mutual_values) - 1):

			if mutual_values[i] > mutual_values[i+1]: 
				minimum_counter = i

			else: break
	

		first_min_mutual_info_sine = minimum_counter + 1

		print 'The first minimum for the mutual information for the sine wave is', first_min_mutual_info_sine

		return first_min_mutual_info_sine


	if data_set == 'amplitude':

		"""Reads in directory and file names and the data set"""

		file_name = "{}/amplitude.dat".format(directory)

		mutual = subprocess.check_output(["mutual", file_name]) 
		print mutual
		
		tempfile = open('mutual.dat', 'w')
		mutual_info_array = np.array(mutual.split()[2:], dtype=float)
		tempfile.write(mutual_info_array)
		tempfile.close()

		idx_odd = range(1, len(mutual_info_array) + 1, 2)
		idx_even = range(0,len(mutual_info_array), 2)

		mutual_delay = mutual_info_array[idx_even]
		mutual_values = mutual_info_array[idx_odd]

		x = mutual_delay
		y = mutual_values
		
		plt.figure(0)
		plt.scatter(x, y)
		
		plt.figure(1)
		plt.plot(x, y)
		plt.show()
		
		minimum_counter = 0

		for i in range(len(mutual_values) - 1):

			if mutual_values[i] > mutual_values[i+1]: 
				minimum_counter = i

			else: break
	

		first_min_mutual_info_amplitude = minimum_counter + 1

		print 'The first minimum for the mutual information for the amplitude data is', first_min_mutual_info_amplitude

		return first_min_mutual_info_amplitude

	#if data_set == 'henon':

		
		

def plot_delays(data_set, resolution, num_delays):

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"
	
	if data_set =='sine':

		"""Reads in directory and file names and the data set"""

		file_name = "{}/test_sine_data_AMPLITUDE.txt".format(directory)

		"""Optimal embedding depends on the application"""

		"""delay and embed routines are an important tool for the visulisation inspection of data"""

		for i in range(num_delays):

			tempfile = open('delay.dat', 'w')
			delay = subprocess.check_output(["delay", file_name, "-d" + str(i + 1)]) 
			tempfile.write(delay)
			tempfile.close()

			x = []
			y = []

			E = open('delay.dat', 'r')
			lines = E.readlines()
			E.close()

			for line in lines:
				templine = line.split()
				x.append(templine[0])
				y.append(templine[1])
			plt.figure(i)
			plt.scatter(x, y)

		plt.show()

	if data_set =='amplitude':
		
		"""Reads in directory and file names and the data set"""

		file_name = "{}/amplitude.dat".format(directory)

		"""Optimal embedding depends on the application"""

		"""delay and embed routines are an important tool for the visulisation inspection of data"""

		
		for j in range(num_delays):

			tempfile = open('delay.dat', 'w')
			delay = subprocess.check_output(["delay", file_name, "-d" + str(j + 1)]) 
			tempfile.write(delay)
			tempfile.close()

			x = []
			y = []

			E = open('delay.dat', 'r')
			lines = E.readlines()
			E.close()

			for line in lines:
				templine = line.split()
				x.append(templine[0])
				y.append(templine[1])
			plt.figure(j)
			plt.scatter(x, y)

		plt.show()

	
	if data_set =='henon':

		file_name = "{}/test_henon_data.txt".format(directory)
		x_coord_file_name = "{}/test_henon_x_coord_data.txt".format(directory)
		y_coord_file_name = "{}/test_henon_y_coord_data.txt".format(directory)

		"""Optimal embedding depends on the application"""

		"""delay and embed routines are an important tool for the visulisation inspection of data"""

		a = np.loadtxt(file_name)

		x = a[0]
		y = a[1]

		plt.figure('Henon map')
		plt.scatter(x, y)

		b = np.loadtxt(x_coord_file_name)

		plt.figure('henon x data')
		plt.scatter(range(len(b)), b)

		c = np.loadtxt(y_coord_file_name)

		plt.figure('henon y data')
		plt.scatter(range(len(c)), c)

		plt.show()	
		

def false_nearest_neighbours(data_set, resolution):

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"

	if data_set =='sine':

		"""Reads in directory and file names and the data set"""

		file_name = "{}/test_sine_data_AMPLITUDE.txt".format(directory)

		delay = find_mutual_information(data_set, resolution)

		false_nearest = subprocess.check_output(["false_nearest", file_name, "-m1", "-M1,10", "-d" + str(delay)]) 

		split_file_name = np.array(false_nearest.split(), dtype = float)
		sine_fnn_embedding= split_file_name[0::4]
		sine_fnn_fraction = split_file_name[1::4]
		plt.figure('fnn vs embedding for sine data')
		plt.axis([1, 10, 0, 1])
		plt.plot(sine_fnn_embedding, sine_fnn_fraction)
		plt.show()

	if data_set =='amplitude':

		"""Reads in directory and file names and the data set"""

		file_name = "{}/amplitude.dat".format(directory)

		delay = find_mutual_information(data_set, resolution)

		false_nearest = subprocess.check_output(["false_nearest", file_name, "-m1", "-M1,30", "-d" + str(delay)]) 

		split_file_name = np.array(false_nearest.split(), dtype = float)
		amp_fnn_embedding= split_file_name[0::4]
		amp_fnn_fraction = split_file_name[1::4]
		plt.figure('fnn vs embedding for amplitude data')
		plt.axis([1, 30, 0, 1])
		plt.plot(amp_fnn_embedding, amp_fnn_fraction)
		plt.show()
		

	if data_set =='henon':

		"""Reads in directory and file names and the data set"""

		#file_name = "{}/test_henon_data.txt".format(directory)

		#false_nearest = subprocess.check_output(["false_nearest", file_name, "-m1", "-M1,5", "-d1"]) 
		#print false_nearest


		
		henon_file_x_coord = "{}/test_henon_x_coord_data.txt".format(directory)
		henon_file_y_coord = "{}/test_henon_y_coord_data.txt".format(directory)
		henon_file_r_coord = "{}/test_henon_r_coord_data.txt".format(directory)

		false_nearest_x = subprocess.check_output(["false_nearest", henon_file_x_coord, "-m1", "-M1,5", "-d1"]) 
		false_nearest_y = subprocess.check_output(["false_nearest",  henon_file_y_coord, "-m1", "-M1,5", "-d1"])
		false_nearest_r = subprocess.check_output(["false_nearest",  henon_file_r_coord, "-m1", "-M1,5", "-d1"]) 

		split_fnn_x = np.array(false_nearest_x.split(), dtype=float)
		split_fnn_y = np.array(false_nearest_y.split(), dtype=float)
		split_fnn_r = np.array(false_nearest_r.split(), dtype=float)
		
		fnn_embedding_x = split_fnn_x[0::4]
		fnn_embedding_y = split_fnn_y[0::4]
		fnn_embedding_r = split_fnn_r[0::4]

		fnn_fraction_x = split_fnn_x[1::4]
		fnn_fraction_y = split_fnn_y[1::4]
		fnn_fraction_r = split_fnn_r[1::4]		

		plt.figure('x-coords')
		plt.plot(fnn_embedding_x, fnn_fraction_x)
		
		plt.figure('y-coords')
		plt.plot(fnn_embedding_y, fnn_fraction_y)

		plt.figure('r-coords')
		plt.plot(fnn_embedding_r, fnn_fraction_r)
		
		plt.show()

def rp(resolution):

	directory = "/local/scratch/sam5g13/Sam_5th-yr_Project/test_data"

	#file_name = "{}/sine_wave3_10Hz.csv".format(directory)

	#file_name = "{}/test_sine_data_resolution_{}.txt".format(directory, resolution)

	time_series = FileReader.file_as_float_array('/local/scratch/sam5g13/Sam_5th-yr_Project/test_data/test_sine_data_AMPLITUDE.txt', column = 0)

	plt.plot(time_series)
	plt.show()

	#Import the time series and specify each parameter 	

	settings = Settings(time_series, embedding_dimension = 1, time_delay = 3, similarity_measure = EuclideanMetric, neighbourhood = FixedRadius(radius = 0.1), min_diagonal_line_length = 2, min_vertical_line_length = 2, min_white_vertical_line_length = 2)

	#Perform the computation and print out the results

	rqacomputation = RQAComputation.create(settings, verbose = True)

	rqaresult = rqacomputation.run()

	print rqaresult

	#Creates the recurrence matrix for viewing

	rpcomputation = RecurrencePlotComputation.create(settings)

	rpresult = rpcomputation.run()

	ImageGenerator.save_recurrence_plot(rpresult.recurrence_matrix, 'sin_recurrence_plot.png')


create_sine_test_data()
#rp(1000)
find_correlation_time('sine', 10)

#Embedding is when the number of false nearest neighbours goes to 0
#false_nearest_neighbours('sine', 1000)
#TISEAN_henon_data()
#find_mutual_information('sine', 100)
#henon_map_data()
#false_nearest_neighbours('sine', 10000)
#henon_map_data()
#plot_delays('henon', 100, 5)
#TISEAN_henon_data()























































