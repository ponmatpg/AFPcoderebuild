import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
import scipy.optimize as spo
import statsmodels.api as sm
from elecModel import elecModel_alt
from datetime import datetime
matplotlib.style.use('ggplot')

directory = os.getcwd()
pepco_data = pd.read_csv(directory + '/data/lmp_data/pepco_price_cleaned.csv')
pepco_data.sort_values(['datetime'],  )
startDate = '2014-01-05'
endDate = '2015-01-05'
# volatility look-back window (30 days, 24 entries per day)
window = 30 * 24

model = elecModel_alt.elecModel()

# fit volatility model and mean_reversion parameters 
model.fit_vol_params(data=pepco_data, startDate=startDate, endDate=endDate, window=window)
model.fit_meanrev_params(data=pepco_data, startDate=startDate, endDate=endDate)
# fit weather correlations 
weather_data = pd.read_csv(directory + '/data/weather_data/DCA_hourly2_clean.csv')

model.fit_weather_correl(price_data=pepco_data, weather_data=weather_data, startDate=startDate, endDate=endDate)

print(model.meanrev_params)
print(model.vol_params)
print(model.weather_correl)


startDate_sim = '2015-01-05'
endDate_sim = '2016-01-05'
T = 1.0
test_prices = list(pepco_data[[True if  startDate <= date <= endDate else False for date in pepco_data['datetime']]]['lmp'])


# reads normal sampling from folder and generates matrix 
# reads sampling csv files in order, stitches together large list of samples 
# changes shape to matrix as needed by elecModel module 
paths = 10
sampling = pd.read_csv(directory + '/simulation/sampling/normal_sampling' + str(paths) + 'paths.csv')
sampling = np.transpose(sampling.as_matrix(columns=[str(x) for x in range(0, paths)]))

print(sampling.shape)

# from normal sampling matrix, calls electricity price model to generate corresponding spot price paths 
no_paths = sampling.shape[0]
path_matrix = model.generateSpotPaths(test_prices[0], T, weather_correl=0.2, N=sampling.shape[1], paths = no_paths, sampling = sampling)
path = list(map(lambda x: np.exp(x / no_paths), map(sum, zip(*path_matrix))))
pd.DataFrame(np.transpose(path_matrix)).to_csv(directory + '/logPricePaths' + str(paths) + '.csv')

# """
# # performs OLS for model and true spot prices 
# test_prices = np.log(test_prices)
# for i in range(paths):
# 	path = path_matrix[i]
# 	temp = pd.DataFrame([np.sum(np.power(np.subtract(test_prices, path[:-2]), 2.0)) / float(len(test_prices))])
# 	temp.to_csv('MSE_logspot_' + str(i) + '.csv')
# 	mod = sm.OLS(test_prices, path[:-2])
# 	est = mod.fit()
# 	print(est.summary())
# """

# path = path_matrix[0]
# plt.plot(test_prices, label='Historical')
# plt.plot(path[:-2], label='Simulated')
# plt.legend()
# plt.title('Simulated path of log Spot Price, 2015-01-05 to 2016-01-05')
# plt.savefig('logSpotSim20152016.png')
# plt.show()
# plt.clf()







# computes mean-square error for volatility model 
# def MSE(data, startDate, endDate, window):
# 	test_data = data[[True if startDate <= date <= endDate else False for date in pepco_data['datetime']]]
# 	test_prices = list(test_data['lmp'])

# 	# fitting normal parameters 
# 	test_logReturns = elecModel.logReturns(test_prices)
# 	test_normal_prices = list(map(lambda x: test_prices[0] * np.exp(x), elecModel.filter_(test_logReturns, threshold=2.5)[0]))
# 	y = elecModel.priceVolSeries(test_normal_prices, window=window)

# 	SSE = 0
# 	dt = 1 / 8760.0
# 	for i in range(len(y)):
# 		SSE += (y[i] - elecModel.sigma(list(vol_params_df['0']), i * dt))**2
# 	MSE = SSE / float(len(y))
# 	return MSE

# # writes to file 
# MSE_df = pd.DataFrame([MSE(pepco_data, startDate, endDate, window)])
# MSE_df.to_csv(directory + '/MSE_sigma.csv')


















