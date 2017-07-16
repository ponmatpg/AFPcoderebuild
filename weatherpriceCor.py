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
pepco_data = pd.read_csv(directory + '/data/lmp_data/pepco_price_cleaned.csv')
pepco_data.sort_values(['datetime'],  )
startDate = '2014-01-05'
endDate = '2015-01-05'

diff_ = 0

weather_data = pd.read_csv(directory + '/data/weather_data/simulation_data.csv')
test_weather = list(weather_data['0'])
test_weather = np.diff(test_weather, diff_)
weather_length = len(test_weather)
test_weather = np.matrix(test_weather)
test_weather.shape = (weather_length, 1)
# test_weather = sm.add_constant(test_weather)

test_data = pepco_data[[True if  startDate <= date <= endDate else False for date in pepco_data['datetime']]]
test_prices = np.log(list(test_data['lmp']))
test_prices = np.diff(test_prices, diff_)
prices_length = len(test_prices)
test_prices = np.matrix(test_prices)
test_prices.shape = (prices_length,1)

mod = sm.OLS(test_prices, test_weather)
est = mod.fit()

print(est.summary())




 










