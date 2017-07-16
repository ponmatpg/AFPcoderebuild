import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
import scipy.optimize as spo
import statsmodels.api as sm
from datetime import datetime
from misc import misc 
matplotlib.style.use('ggplot')


class elecModel(object):
	def __init__(self, meanrev_params=None, vol_params=None, weather_correl = 1.0):
		self.meanrev_params = meanrev_params
		self.vol_params = vol_params
		self.weather_correl = weather_correl

	def fit_meanrev_params(self, data, startDate, endDate, dt = 1.0/ (365 * 24)):
		test_data = data[[True if startDate <= date <= endDate else False for date in data['datetime']]]
		test_prices = list(test_data['lmp'])
		log_spot_prices = list(map(np.log, test_prices))

		delta_xts = np.transpose(np.matrix([float(log_spot_prices[i+1]) - log_spot_prices[i] for i in range(0, len(log_spot_prices) - 1)]))
		xts = log_spot_prices[:-1]
		xts_with_const = np.transpose(np.matrix([[1] * len(xts)] + [xts]))

		mod = sm.OLS(delta_xts, xts_with_const)
		res = mod.fit()

		a0, a1 =  res.params
		a = -1 * a1 / dt 
		xbar = a0 / (a * dt)

		self.meanrev_params = [a, xbar]

	# periodic functional form for our seasonal volatility 
	def sigma(self, params, t):
		pi = 4 * np.arctan(1)
		output = 0
		for i in range(0, len(params)):
			output += params[i] * np.cos(2*pi*i*(t))

		return output 

	def fit_vol_params(self, data, startDate, endDate, window):

		# cost function used for fitting parameters for the seasonal volatility function 
		def cost(params, t, y):
			return sum([(self.sigma(params, t) - y)**2 for t,y in zip(t,y)])

		test_data = data[[True if startDate <= date <= endDate else False for date in data['datetime']]]
		test_prices = list(test_data['lmp'])

		# fitting normal parameters 
		test_logReturns = misc.logReturns(test_prices)
		test_normal_prices = list(map(lambda x: test_prices[0] * np.exp(x), misc.filter_(test_logReturns, threshold=2.5)[0]))
		y = misc.priceVolSeries(test_normal_prices, window=window)
		dt = 1 / (365.0 * 24.0)
		t = [dt * i for i in range(len(y))]

		res = spo.minimize(cost, x0=[0] * 13, args=(t,y))
		
		self.vol_params = list(res.x)

	def fit_weather_correl(self, price_data, weather_data, startDate, endDate):
		# take weather model, subtract off the contribution from the ARIMA portion
		# then find correlation with the spot electricity prices from the test data 

		price_data.sort_values(['datetime'],  inplace=True)
		weather_data.sort_values(['Date', 'Time'], ascending=[True, True], inplace=True)

		weather_data['Date'] = list(map(lambda x: datetime.strptime(str(x), '%Y%m%d'), weather_data['Date']))
		weather_data['Date'] = list(map(lambda x: x.strftime('%Y-%m-%d'), weather_data['Date']))

		# print(weather_data)

		diff_ = 0
		test_weather_data = weather_data[[True if startDate <= date <= endDate else False for date in weather_data['Date']]]
		test_weather = list(test_weather_data['DryBulbFarenheit'])
		test_weather = np.diff(test_weather, diff_)
		weather_length = len(test_weather)
		# test_weather = sm.add_constant(test_weather)

		test_data = price_data[[True if  startDate <= date <= endDate else False for date in price_data['datetime']]]
		test_prices = np.log(list(test_data['lmp']))
		test_prices = np.diff(test_prices, diff_)
		prices_length = len(test_prices)

		length = min(weather_length, prices_length)
		test_weather = np.matrix(test_weather[:length])
		test_prices = np.matrix(test_prices[:length])
		test_weather.shape = (length, 1)
		test_prices.shape = (length,1)

		mod = sm.OLS(test_prices, test_weather)
		est = mod.fit()

		weather_correl = est.params[0]






	def generateSpotPaths(self, S_0, T, weather_correl=1.0, jump_params = None, N = 1000, paths = 1000, sampling = None):

		x_0 = np.log(S_0)
		a, xbar = self.meanrev_params
		dt = T / float(N)

		# for each desired path of spot prices, we generate and average 'paths_per_path' many paths to generate a single desired path 
		paths_per_path = 1
		if sampling is None:
			sampling = np.matrix(np.random.normal(size = N * paths))
			sampling.shape = (paths, N)
		

		sampling_2 = np.random.normal(size= sampling.shape[0] * sampling.shape[1] * paths_per_path)
		sampling_2 = np.matrix(sampling_2)
		sampling_2.shape = (sampling.shape[0] * paths_per_path, sampling.shape[1])

		final_sampling = [None] * sampling.shape[0] * sampling.shape[1] * paths_per_path
		final_sampling = np.matrix(final_sampling)
		final_sampling.shape = (sampling.shape[0] * paths_per_path, sampling.shape[1])

		for i in range(final_sampling.shape[0]):
			for j in range(final_sampling.shape[1]): 
				weather_sample = sampling[int(i / paths_per_path), j]
				final_sampling[i, j] = self.weather_correl * weather_sample + np.sqrt(1 - self.weather_correl**2) * sampling_2[i,j]

		if jump_params is not None:
			phi, kappa_bar, gamma = jump_params	

		path_output = [[None for i in range(N+1)] for i in range(paths * paths_per_path)]
		for i in range(paths * paths_per_path):
			path_output[i][0] = x_0

		for i in range(paths * paths_per_path):
			print(i)
			for j in range(1, N+1):
				# jump = kappa_bar + gamma * sampling_2[i, j-1] if sampling_unif[i,j-1] < phi * dt else 0
				dxt = a * (xbar - path_output[i][j-1]) * dt + sigma(self.vol_params, dt* j ) * final_sampling[i, j-1] * np.sqrt(dt) #+ jump
				path_output[i][j] = path_output[i][j-1] + dxt
		
		f = lambda lst: [sum(lst[paths_per_path*i:paths_per_path* (i+1)]) / float(paths_per_path) for i in range(len(lst) / paths_per_path)]
		path_output = list(zip(*map(f, zip(*path_output))))

		return path_output