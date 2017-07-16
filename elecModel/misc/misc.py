import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
import scipy.optimize as spo
import statsmodels.api as sm
matplotlib.style.use('ggplot')

# computes log returns given list of spot electricity prices 
def logReturns(lst):
	return list(map(np.log, [lst[i+1]/lst[i] for i in range(0, len(lst) - 1)]))

# if data has NaN values (which sometimes occurrs, fills in with simple linearly interpolated values)
def linearly_interpolate_nans(y):
	if all([True if x is None else False for x in y]):
		return None

	idx_1 = min([i for i in range(len(y)) if y[i] is not None])
	y[:idx_1] = [y[idx_1]] * idx_1
	
	idx_2 = max([i for i in range(len(y)) if y[i] is not None])
	y[idx_2:] = [y[idx_2]] * (len(y)  - idx_2)

	bd1 = [i for i in range(len(y) - 1) if y[i+1] is None and y[i] is not None]
	bd2 = [i for i in range(1, len(y)) if y[i-1] is None and y[i] is not None]
	bounding_pairs = [(a,b) for a,b  in zip(bd1, bd2)]

	for pair in bounding_pairs:
		idx_1, idx_2 = pair

		slope = (y[idx_2] - y[idx_1]) / float(idx_2 - idx_1)
		for i in range(idx_1, idx_2):
			y[i] = y[idx_1] + (i - idx_1) * slope
	return y

# deprecated, was intended for a jump diffusion model where we would have to separate 'normal' and 'jump' data points 
def filter_(logReturns, threshold):
	filtered_normal = [logRet if abs(logRet) < threshold else None for logRet in logReturns]
	filtered_jumps = [logRet if abs(logRet) >= threshold else None for logRet in logReturns]

	filtered_normal = linearly_interpolate_nans(filtered_normal)

	return filtered_normal, filtered_jumps

# computes spot price volatility for a given list of prices
def priceVol(prices, factor=np.sqrt(365* 24)):
	priceReturns = [prices[i] / float(prices[i+1]) for i in range(1, len(prices) - 1)]
	logPriceReturns = list(map(np.log, priceReturns))

	return np.std(logPriceReturns) * factor 

# computes rolling window price volatility given a list of spot prices 
def priceVolSeries(test_prices, window = 30 * 24, factor = np.sqrt(365*24)):
	vols = [None] * (len(test_prices) - window)
	for i in range(window, len(test_prices)):
		prices = test_prices[i-window:i]
		vols[i-window] = priceVol(prices, factor=factor)
	return vols


