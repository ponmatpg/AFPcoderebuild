# import matplotlib.pyplot as plt 
# from statsmodels.tsa.arima_process import arma_generate_sample
# import numpy as np

# hourly_ar = [0.5] * 3
# hourly_ma = [0.5] * 6
# hourly_samples = 10000
# hourly_sigma = 1.0

# temp = arma_generate_sample(ar=hourly_ar, ma=hourly_ma, nsample=hourly_samples,sigma=hourly_sigma) 


# hourly_ar = np.multiply(-1.0, hourly_ar)



""" This function is almost exactly the same as statsmodels.tsa.arima_process.arma_generate_sample. 

The generation of an ARMA process requires drawing from a standard normal r.v.; we needed to generate an ARMA process and 
have access to the underlying normal samples. Since the function from statsmodels is a black box that only outputs the 
ARMA samples, we wrote our own function to a. replicate the ARMA sample generation, b. also output the normal sampling 


ar - ar coefficients 
ma - ma coefficients
nsample - number of samples 
sigma - stdev
write_sample - bool for writing normal sampling to csv 
"""






def arma_generate_sample_and_noise(ar, ma, nsample, sigma=1.0, write_sample=False):
	import numpy as np
	import pandas as pd 
	import os

	ar = np.multiply(-1, ar)
	p, q = len(ar)-1, len(ma)-1

	sampling = np.random.randn(q + nsample,)
	if write_sample:
		i = 0
		while os.path.exists(os.getcwd() + '/sampling/sampling' + str(i) + '.csv'):
			i+=1 
		pd.DataFrame(sampling).to_csv(os.getcwd() + '/sampling/sampling' + str(i) + '.csv')

	scaled_sampling = list(map(lambda x: x * sigma**2, sampling))
	arma_sample = [0] * p + [None] * nsample

	for i in range(p, p + nsample):
		arma_sample[i] = (sum(np.multiply(scaled_sampling[i-p:i-p+q+1], ma[::-1])) + sum(np.multiply(arma_sample[i-p:i], ar[1:][::-1]))) 
		arma_sample[i] /= -1*ar[0]
	return arma_sample[p:], sampling


""" FOLLOWING COMMENTED OUT CODE IS FOR TESTING PURPOSES """
"""
import statsmodels.tsa.arima_model 
import numpy as np
np.random.seed(12345)
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
y = arma_generate_sample_and_noise(ar,ma,10000, 1.0)[0]
model = statsmodels.tsa.arima_model.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
print(model.params) """

