import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
import scipy.optimize as spo
import statsmodels.api as sm
from elecModel import elecModel
matplotlib.style.use('ggplot')



directory = os.getcwd()
pepco_data = pd.read_csv(directory + '/pepco_price_cleaned.csv')

# plotting function for a graph we needed 
def plotting(data):
	startDate = '2010-01-05'
	endDate = '2015-01-05'
	
	data = data[[True if  startDate <= date <= endDate else False for date in data['datetime']]]
	lmp = list(data['lmp'])
	plt.plot(lmp)
	plt.ylabel('Spot LMP ($/MWh)')
	plt.xticks([w*8760 for w in range(6)],['200%i'%w for w in range(6)])
	plt.xlabel('Year')
	plt.savefig('spotprices_2010_2015.png')
	plt.clf()

	day_lmp = lmp[:24]
	plt.plot(day_lmp)
	plt.ylabel('Spot LMP ($/MWh)')
	plt.xticks(list(range(0,25,6)),['12am', '6am', '12pm', '6pm', '12am'])
	plt.xlabel('Hour')
	plt.savefig('hourly_prices_2010_01_05.png')
	plt.clf()

# plotting(pepco_data)



startDate = '2015-01-05'
endDate = '2016-01-05'
T = 1.0
test_data = pepco_data[[True if  startDate <= date <= endDate else False for date in pepco_data['datetime']]]
test_prices = list(test_data['lmp'])


# reads volatility and mean reversion parameters 
vol_params = pd.read_csv(directory + '/vol_params.csv')['0']
meanrev_params = pd.read_csv((directory + '/meanrev_params.csv'))['0']

# reads normal sampling from folder and generates matrix 
# reads sampling csv files in order, stitches together large list of samples 
# changes shape to matrix as needed by elecModel module 
sampling = []
paths = 10
for i in range(paths):	
	sample = pd.read_csv(directory + '/sampling/sampling' + str(i) + '.csv')
	sampling += list(sample[sample.columns.values[1]])
sampling_length = len(sampling)
sampling = np.matrix(sampling)
sampling.shape = (paths, sampling_length / paths)


# from normal sampling matrix, calls electricity price model to generate corresponding spot price paths 
no_paths = sampling.shape[0]
path_matrix = elecModel.generateSpotPaths(test_prices[0], meanrev_params, vol_params, T, weather_correl=0.2, N=sampling.shape[1], paths = no_paths, sampling = sampling)
# path = list(map(lambda x: np.exp(x / no_paths), map(sum, zip(*path_matrix))))
pd.DataFrame(np.transpose(path_matrix)).to_csv(directory + '/logPricePaths' + str(paths) + '.csv')

"""
# performs OLS for model and true spot prices 
test_prices = np.log(test_prices)
for i in range(paths):
	path = path_matrix[i]
	temp = pd.DataFrame([np.sum(np.power(np.subtract(test_prices, path[:-2]), 2.0)) / float(len(test_prices))])
	temp.to_csv('MSE_logspot_' + str(i) + '.csv')
	mod = sm.OLS(test_prices, path[:-2])
	est = mod.fit()
	print(est.summary())
"""

path = path_matrix[0]
plt.plot(test_prices, label='Historical')
plt.plot(path[:-2], label='Simulated')
plt.legend()
plt.title('Simulated path of log Spot Price, 2015-01-05 to 2016-01-05')
plt.savefig('logSpotSim20152016.png')
plt.show()
plt.clf()






