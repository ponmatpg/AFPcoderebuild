import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
import scipy.optimize as spo
import statsmodels.api as sm
from elecModel import elecModel_alt
matplotlib.style.use('ggplot')


# directory = os.getcwd()
# pepco_data = pd.read_csv(directory + '/pepco_price_cleaned.csv')

# # plotting function for a graph we needed 
# def plotting(data):
# 	startDate = '2010-01-05'
# 	endDate = '2015-01-05'
	
# 	data = data[[True if  startDate <= date <= endDate else False for date in data['datetime']]]
# 	lmp = list(data['lmp'])
# 	plt.plot(lmp)
# 	plt.ylabel('Spot LMP ($/MWh)')
# 	plt.xticks([w*8760 for w in range(6)],['200%i'%w for w in range(6)])
# 	plt.xlabel('Year')
# 	plt.savefig('spotprices_2010_2015.png')
# 	plt.clf()

# 	day_lmp = lmp[:24]
# 	plt.plot(day_lmp)
# 	plt.ylabel('Spot LMP ($/MWh)')
# 	plt.xticks(list(range(0,25,6)),['12am', '6am', '12pm', '6pm', '12am'])
# 	plt.xlabel('Hour')
# 	plt.savefig('hourly_prices_2010_01_05.png')
# 	plt.clf()

# # plotting(pepco_data)