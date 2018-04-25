"""
BLOCK AVERAGING SCRIPT THAT FINDS CORRELATION TIME


Notes
-----

Contains all the functions that are needed to block average the
surface tension time series and then find the correlation time.
"""
#from __future__ import division
import subprocess, os, sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats

def get_block_error(ST, directory, file_name, ntb, ow_ntb=False):

	"""

	This function return the correlation time and also the error adjusted using the correlation time. Its main purpose is to save time by allowing us to start 

	Parameters 
	----------

	ST : array
		Array contain the values of the surface tension for that specific file

	directory: pathway
		The file tree that shows where the file is stored

	file_name: Name
		The name of the file that the function runs over

	ntb: Integer
		Final tb value

	ow_ntb: boolian 
		If TRUE calculate new value, if FALSE us previously calculated value of ntb

	"""

	if not os.path.exists('{}/surface_ten_data'.format(directory)): os.mkdir('{}/surface_ten_data'.format(directory)) #If the file doesn't exist make it

	if not ow_ntb: #If ow_ntb is TRUE, if the parameter has been changed when inputted in the function, then try the following
		try:
			with file('{}/surface_ten_data/{}_{}_PT.npy'.format(directory, file_name, ntb), 'r') as infile: #Read this file and extract data
				pt_st = np.load(infile) #Loading in the data as pt_st
		except: ow_ntb = True # If the previous step doesn't work then set ow_ntb to TRUE. The file must be over written.

	if ow_ntb: # If ow_ntb is TRUE then new values are produced
		old_pt_st, old_ntb = load_pt('{}/surface_ten_data'.format(directory)) # Load in as much data as we already have (this is to save time)
		#print 'This is old_pt_st and old_ntb', old_pt_st, old_ntb

		if old_ntb == 0 or ow_ntb: # If there is no previous data then run the block error function and create the data from scratch
			pt_st = block_error(ST, ntb) # run the function to get the block errors from the surface tension
			#print 'if old_ntb = 0 then pt_st is', pt_st

		elif old_ntb > ntb: # If old_ntb has more information than what we need, only load in the surface tension information up to the point we want
			with file('{}/surface_ten_data/{}_{}_PT.npy'.format(directory, file_name, old_ntb), 'r') as infile: # loading the file in
                        	pt_st = np.load(infile) #Assigning a variable name
			pt_st = pt_st[:ntb] #Slicing the data so we only store the amount we need

		elif old_ntb < ntb: # If the data we have already made is correct but incomplete, load in the data up to the point we have 
			with file('{}/surface_ten_data/{}_{}_PT.npy'.format(directory, file_name, old_ntb), 'r') as infile: # Load in the data
                        	pt_st = np.load(infile) #Assigning a variable name

			new_pt_st = block_error(ST, ntb) # Creating the missing data and call it new_pt_st
			pt_st = np.concatenate(pt_st, new_pt_st) # Add the old data and the new data into one long data det
		
		with file('{}/surface_ten_data/{}_{}_PT.npy'.format(directory, file_name, ntb), 'w') as outfile: # Use whichever of the above files we have bought forward
			np.save(outfile, pt_st) # Save the file

	corr_time_st = get_corr_time(pt_st, ntb) # Using the get corr time function on the data set created above. Assigning it a variable
	

	m_err_st = (np.std(ST) * np.sqrt(corr_time_st / len(ST))) # Calculating the error corrected using the correlation time

        return corr_time_st, m_err_st # Return the correlation time and the error	


def load_pt(directory):

	"""

	This function loads in the surface tension data that is currently available

	Parameters 
	----------

	directory: pathway
		The file tree that shows where the file is stored

	"""

	proc = subprocess.Popen('ls {}/*PT.npy'.format(directory), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Pipeing commands, lists all the files with PT.npy at the end of the file name
	out, err = proc.communicate() # Assigning variable names to the piped command
	pt_files = out.split() # Splits the output line by line

	length = len(directory) + 1 # length is length + 1 so that it includes the forward slash
	pt_st = [] # Empty array so that if there is no diretory it still returns the empty list
	ntb = 0 # set ntb = 0 to begin with
	ntb_file_name = '' # Empty string 
	
	for file_name in pt_files: # runs through all the pt files
		temp = file_name[length:-4].split('_') # Splits the PT.npy part up with underscores
		if int(temp[-2]) >= ntb: # If the value of the thrid from last character from the file is greater than or equal to ntb then we need to reassign below
			ntb_file_name = file_name # Assigning a variable name
			ntb = int(temp[-2]) # ntb is the 3rd from last 
			
	try: 
		with file(ntb_file_name, 'r') as infile:
			pt_st = np.load(infile) #Try loading the surface tension data file
	except: pass # If it can't be loaded pass

	return pt_st, ntb # Returns the surface tension and the number of blocks


def block_error(A, ntb, s_ntb=2):

	"""

	This function finds the error of the time series using the block averaging method. Smallest blocksize is length 2.

	Parameters 
	----------

	A : array
		Array contain the values of the surface tension for that specific file

	ntb: Integer
		Final tb value

	s_ntb: Integer
		Initial tb value

	"""
	
	var2 = np.var(A) # Variance of the array A
	pt = [] # Empty array

	for tb in range(s_ntb, ntb): 
		stdev1 = np.var(blocksav(A, tb)) # Assigning a variable name to the block average
		pt.append(stdev1 * tb / var2) # Append to the empty array the standard deviation of the block average multiplied by tb dividied by the variance of the array A

	print 'np.array(pt) is', np.array(pt)

	return np.array(pt) # Returns the block error of the array A


def blocksav(A, tb):

	"""

	Returns the average of the blocks

	Parameters 
	----------

	A : array
		Array contain the values of the surface tension for that specific file

	tb: Integer
		Size of each block

	"""

	nb = len(A)/tb # nb is the number of whole blocks, tb is block size.... nb cycles through each block 
	blocks = np.zeros(nb) # Set up a matrix of zeros the same shape as the array nb
	for i in range(nb): # Ranges over the number of blocks, so uses each block in turn
		blocks[i] += np.mean(A[i*tb: (i+1)*tb]) # Set each indici of the blocks matrix equal to the average the original array * block size until the next bin along

	return blocks # Returns blocks average


def get_corr_time(pt, ntb):

	"""

	Get the correlation time using the block error and the number of steps in each block

	Parameters 
	----------

	pt : array
		Array contain the values of the surface tension for that specific file

	ntb: Integer
		Final tb value

	"""

	x = [(1./tb) for tb in range(2,ntb)] # x array for the linear regression to be performed upon
        y = [(1./pt[i]) for i in range(ntb-2)] # y array for the linear regression to be perfomed upon

	cut_tb = int(len(x) * 1. / 2) #Reducing the array length to get rid of unwanted extra info after convergence has been reached.
	x = x[:cut_tb]
	y = y[:cut_tb]

	"""Plotting the x and y coords"""

	plt.figure('1/tb vs 1/pt')
	plt.xlabel('1/tb')
	plt.ylabel('1/pt')
	plt.plot(x, y)

	plt.figure('tb vs pt')
	plt.xlabel('tb')
	plt.ylabel('pt')
	plt.plot(np.divide(1, x), np.divide(1, y))

        m, intercept, _, _, _ = stats.linregress(x,y) # Getting the slope and the intercept

	return 1. / intercept # Returning the reciprical of the intercept


def ARGON_plot_correlations_v_temp(directory):

	"""

	Plots the correlation time against the temperature for Argon

	Parameters 
	----------

	directory : string
		The file path for where the file is stored

	"""

	temp_range = np.arange(85,146,5) # The temperatures that each simulation is run at
	cutoff_range = [22] # The cutoff range the simulation is run at, no correction terms are implemented
	corr_time_array = [] # Array to store the correlation time of each temperature
	correlated_error_array = [] # Array to store the correlated error of each temperature

	for temp in temp_range:
	
		"""For loop ranges over each different temperature file"""
	
		file_name = 'argon_{}_22_6000_ST'.format(temp) #All file names are named in this exact format
	
		with open('{}/surface_ten_data/{}.npy'.format(directory, file_name), 'r') as infile:
			surf_ten = np.load(infile)

		ntb = len(surf_ten) / 100 # Maximum block size 
		print len(surf_ten)

		correlation_time, correlated_error = get_block_error(surf_ten, directory, file_name, ntb, False) # Function that return correlation time and error adjusted for the correlation time

		corr_time_array.append(correlation_time * 10) # Appending correlation time from each temperature to an array in sample units. This is every 10 time steps, so need to times by 10.
		correlated_error_array.append(correlated_error) # Appending correlated error from each temperature to an array, dont need to adjust as in units of samples

	#print corr_time_array, correlated_error_array


	"""Plotting the correlation times against temperature"""

	plt.figure('ARGON - correlation time vs temperature')
	plt.xlabel('Temperature range / K')
	plt.ylabel('Correlation time')

	plt.plot(temp_range, corr_time_array)

	"""Plotting the correlated errors against temperature"""

	plt.figure('ARGON - correlated error vs temperature')
	plt.xlabel('Temperature range / K')
	plt.ylabel('Correlated error')

	plt.plot(temp_range, correlated_error_array)

	return corr_time_array, correlated_error_array

directory = '/local/scratch/sam5g13/AMBER/ARGON'

corr_times, _ = ARGON_plot_correlations_v_temp(directory)

print corr_times
plt.show()































