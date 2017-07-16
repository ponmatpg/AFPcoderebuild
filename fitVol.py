import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
import scipy.optimize as spo
from elecModel import elecModel
matplotlib.style.use('ggplot')

directory = os.getcwd()
pepco_data = pd.read_csv(directory + '/pepco_price_cleaned.csv')
pepco_data.sort_values(['datetime'],  )
startDate = '2013-12-05'
endDate = '2015-01-05'
window = 30 * 24

# fitting of seasonal volatility model (i.e. finding mean reversion and volatility parameters) 
# just calls other function for mean reversion parameters 
# for volatility parameters we minimize elecModel.cost 
def fitModel(data, startDate, endDate, window):
	test_data = data[[True if startDate <= date <= endDate else False for date in pepco_data['datetime']]]
	test_prices = list(test_data['lmp'])
	meanrev_params = elecModel.fitSpotParams(test_prices)

	# fitting normal parameters 
	test_logReturns = elecModel.logReturns(test_prices)
	test_normal_prices = list(map(lambda x: test_prices[0] * np.exp(x), elecModel.filter_(test_logReturns, threshold=2.5)[0]))
	y = elecModel.priceVolSeries(test_normal_prices, window=window)
	dt = 1 / (365.0 * 24.0)
	t = [dt * i for i in range(len(y))]
	print('begin')

	res = spo.minimize(elecModel.cost, x0=[0] * 13, args=(t,y))
	vol_params = list(res.x)

#	plt.plot(y)
#	plt.plot(list(map(lambda t: elecModel.sigma(vol_params, t), t)))
#	plt.title('30-day Volatility of PEPCO Spot Price')
#	plt.savefig('30dayVolFitted.jpg')
#	plt.show()

	return vol_params, meanrev_params

# writing params to csv
vol_params_df, meanrev_params_df = list(map(lambda lst: pd.DataFrame(np.array(lst)), fitModel(pepco_data, startDate, endDate, window)))
vol_params_df.to_csv(directory + '/vol_params.csv')
meanrev_params_df.to_csv(directory + '/meanrev_params.csv')


vol_params_df = pd.read_csv(directory + '/vol_params.csv')


# computes mean-square error for volatility model 
def MSE(data, startDate, endDate, window):
	test_data = data[[True if startDate <= date <= endDate else False for date in pepco_data['datetime']]]
	test_prices = list(test_data['lmp'])

	# fitting normal parameters 
	test_logReturns = elecModel.logReturns(test_prices)
	test_normal_prices = list(map(lambda x: test_prices[0] * np.exp(x), elecModel.filter_(test_logReturns, threshold=2.5)[0]))
	y = elecModel.priceVolSeries(test_normal_prices, window=window)

	SSE = 0
	dt = 1 / 8760.0
	for i in range(len(y)):
		SSE += (y[i] - elecModel.sigma(list(vol_params_df['0']), i * dt))**2
	MSE = SSE / float(len(y))
	return MSE

# writes to file 
MSE_df = pd.DataFrame([MSE(pepco_data, startDate, endDate, window)])
MSE_df.to_csv(directory + '/MSE_sigma.csv')


