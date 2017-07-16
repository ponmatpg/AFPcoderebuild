# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:41:08 2017

@authors: Hao Li, Paul Ponmattam
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
import sklearn.linear_model as linear_model
from armaSampling import armaSampling 
from datetime import datetime
import os 

### I cleaned the data in R, and use cleaned data here "DCA_hourly2_clean.csv" here
data=pd.read_csv(os.getcwd() + "/data/weather_data/DCA_hourly2_clean.csv")
data.sort_values(['Date', 'Time'], ascending=[True, True], inplace=True)


#Time_list=(52,152,252,352,452,552,652,752,852,952,1052,1152,1252,1352,1452,1552,1652,1752,1852,1952,2052,2152,2252,2352)
#data = data[np.in1d(data["Time"], Time_list)]
#data=data[["Date","Time","DryBulbFarenheit"]]

temperature=np.array(data["DryBulbFarenheit"])
################################################# yearly trend
yearly_tem=np.zeros(5)
i=0
while(i<len(yearly_tem)):
    yearly_tem[i]=np.mean(temperature[(i*len(temperature)/5):((i+1)*len(temperature))/5])   
    i=i+1
plt.plot(yearly_tem)                     
## only 5 year data...not enough to distingguish trend... but actually no trend even if use 10+ years data (another dataset)
regr = linear_model.LinearRegression()

################################################# daily cycle within a year
daily_tem=np.zeros(len(temperature)/24)
i=0
while(i<len(daily_tem)):
    daily_tem[i]=0.5*(max(temperature[i*24:(i+1)*24])+min((temperature[i*24:(i+1)*24])))
    i=i+1
plt.plot(daily_tem)
daily_fft=np.fft.fft(daily_tem)
daily_fft_energy=abs(daily_fft)
plt.plot(daily_fft_energy)

daily_explain=daily_fft.copy()
daily_fft_explain=daily_fft_energy.copy()
daily_explain[np.argmax(daily_fft_explain)]=0
daily_fft_explain[np.argmax(daily_fft_explain)]=0
daily_explain[np.argmax(daily_fft_explain)]=0
daily_fft_explain[np.argmax(daily_fft_explain)]=0
daily_explain[np.argmax(daily_fft_explain)]=0
daily_fft_explain[np.argmax(daily_fft_explain)]=0

daily_explain=daily_fft-daily_explain
daily_fft_explain=daily_fft_energy-daily_fft_explain
plt.plot(daily_fft_explain)
daily_explain=(np.fft.ifft(daily_explain)).real
plt.plot(daily_explain)

daily_fft_rsquare=1-sum((daily_tem-daily_explain)**2)/sum((daily_tem-np.mean(daily_tem))**2)
sum(daily_fft_explain[1:len(daily_fft_explain)]**2)/sum(daily_fft_energy[1:len(daily_fft_energy)]**2)

daily_residual=daily_tem-daily_explain
plt.plot(daily_residual)

plt.plot(acf(daily_residual))
plt.plot(pacf(daily_residual))
plt.plot(np.zeros(len(acf(daily_residual))))

daily_model=ARIMA(daily_residual,order=(1,0,1))
fit_daily_residual=daily_model.fit(disp=-1).fittedvalues
daily_arima_rsquare=np.corrcoef(fit_daily_residual,daily_residual)[1,0] **2

daily_total_explain=daily_explain+fit_daily_residual
daily_total_rsquare=1-sum((daily_tem-daily_total_explain)**2)/sum((daily_tem-np.mean(daily_tem))**2)

plt.plot(daily_tem)
plt.plot(daily_total_explain)
############################################### amplitude for intraday cycle
amplitude=np.zeros(len(temperature)/24)
i=0
while(i<len(amplitude)):
    amplitude[i]=max(temperature[(i*24):(i*24+24)])-min(temperature[(i*24):(i*24+24)])
    i=i+1
plt.plot(amplitude)

amplitude_fft=np.fft.fft(amplitude)
amplitude_fft_energy=abs(amplitude_fft)
plt.plot(amplitude_fft_energy)

amplitude_explain=amplitude_fft.copy()
amplitude_fft_explain=amplitude_fft_energy.copy()
amplitude_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_fft_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_fft_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_fft_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_fft_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_explain[np.argmax(amplitude_fft_explain)]=0
amplitude_fft_explain[np.argmax(amplitude_fft_explain)]=0

amplitude_explain=amplitude_fft-amplitude_explain
amplitude_fft_explain=amplitude_fft_energy-amplitude_fft_explain
plt.plot(amplitude_fft_explain)
amplitude_explain=(np.fft.ifft(amplitude_explain)).real
plt.plot(amplitude_explain)

amplitude_residual=amplitude-amplitude_explain

amplitude_model=ARIMA(amplitude_residual,order=(1,0,1))


################################################ hourly cycle within a day
hourly_tem=temperature.copy()
plt.plot(hourly_tem)
i=0
while(i<len(hourly_tem)):
    hourly_tem[i]=hourly_tem[i]-daily_tem[i//24]
    hourly_tem[i]=hourly_tem[i]/amplitude[i//24]
    i=i+1
plt.plot(hourly_tem)
hourly_fft=np.fft.fft(hourly_tem)
hourly_fft_energy=abs(hourly_fft)
plt.plot(hourly_fft_energy)

hourly_explain=hourly_fft.copy()
hourly_fft_explain=hourly_fft_energy.copy()
hourly_explain[np.argmax(hourly_fft_explain)]=0
hourly_fft_explain[np.argmax(hourly_fft_explain)]=0
hourly_explain[np.argmax(hourly_fft_explain)]=0
hourly_fft_explain[np.argmax(hourly_fft_explain)]=0
hourly_explain[np.argmax(hourly_fft_explain)]=0
hourly_fft_explain[np.argmax(hourly_fft_explain)]=0
hourly_explain[np.argmax(hourly_fft_explain)]=0
hourly_fft_explain[np.argmax(hourly_fft_explain)]=0

hourly_explain=hourly_fft-hourly_explain
hourly_fft_explain=hourly_fft_energy-hourly_fft_explain
plt.plot(hourly_fft_explain)
hourly_explain=(np.fft.ifft(hourly_explain)).real
plt.plot(hourly_explain)

hourly_fft_rsquare=1-sum((hourly_tem-hourly_explain)**2)/sum((hourly_tem-np.mean(hourly_tem))**2)

hourly_residual=hourly_tem-hourly_explain
plt.plot(hourly_residual)

plt.plot(acf(hourly_residual))
plt.plot(pacf(hourly_residual))
plt.plot(np.zeros(len(acf(hourly_residual))))

hourly_model=ARIMA(hourly_residual,order=(1,0,1))
fit_hourly_residual=hourly_model.fit(disp=-1).fittedvalues
hourly_arima_rsquare=np.corrcoef(fit_hourly_residual,hourly_residual)[1,0] **2
';'
hourly_total_explain=hourly_explain+fit_hourly_residual
hourly_total_rsquare=1-sum((hourly_tem-hourly_total_explain)**2)/sum((hourly_tem-np.mean(hourly_tem))**2)

plt.plot(hourly_tem[1:240])
plt.plot(hourly_total_explain[1:240])
####################### total estimation #############
final_estimation=np.zeros(len(temperature))
i=0
while(i<len(temperature)):
    final_estimation[i]=hourly_total_explain[i]*amplitude_explain[i//24]+daily_total_explain[i//24]
    i=i+1
plt.plot(final_estimation)
total_rsquare=1-sum((final_estimation-temperature)**2)/sum((temperature-np.mean(temperature))**2)

###########################################################
######################## simulation #######################
hourly_ar=np.r_[1, hourly_model.fit(disp=0).arparams]
hourly_ma=np.r_[1,hourly_model.fit(disp=0).maparams]
hourly_samples=365*24
hourly_sigma=np.sqrt(0.01369)
daily_ar=np.r_[1, daily_model.fit(disp=0).arparams]
daily_ma=np.r_[1,daily_model.fit(disp=0).maparams]
daily_samples=365
daily_sigma=np.sqrt(25.18)
amplitude_ar=np.r_[1, amplitude_model.fit(disp=0).arparams]
amplitude_ma=np.r_[1,amplitude_model.fit(disp=0).maparams]
amplitude_samples=365
amplitude_sigma=np.sqrt(28.86)

simulation=np.array([])
paths = 10
k=0
sampling = [[None]] * (paths + 1)
while(k<paths):
    hourly_residual_simulation, sampling[k] = armaSampling.arma_generate_sample_and_noise(ar=hourly_ar, ma=hourly_ma, nsample=hourly_samples,sigma=hourly_sigma, write_sample=False)
    
    # arma_generate_sample(ar=hourly_ar, ma=hourly_ma, nsample=hourly_samples,sigma=hourly_sigma)   
    daily_residual_simulation=arma_generate_sample(ar=daily_ar, ma=daily_ma, nsample=daily_samples,sigma=daily_sigma)   
    amplitude_residual_simulation=arma_generate_sample(ar=amplitude_ar, ma=amplitude_ma, nsample=amplitude_samples,sigma=amplitude_sigma)
    i=0   
    path=np.zeros(365*24)
    while (i<len(hourly_residual_simulation)):
        path[i]=(hourly_explain[i]+hourly_residual_simulation[i])*(amplitude_explain[i//24]+amplitude_residual_simulation[i//24])+daily_residual_simulation[i//24]+daily_explain[i//24]
        i=i+1
    if(k==0):
        simulation=path.copy()
    else:
        simulation=np.c_[simulation,path]
    k=k+1

sampling = zip(*sampling)
pd.DataFrame(sampling).to_csv(os.getcwd() + '/simulation/sampling/normal_sampling' + str(paths) + 'paths.csv')

#### compare the actual and the simulation data, one year, 365 *24 hour
plt.plot(temperature[0*365*24:1*365*24])
plt.plot(final_estimation[1*365*24:2*365*24])
plt.plot(simulation[:,1])

df=pd.DataFrame(simulation)
df.to_csv(os.getcwd() + "/simulation/simulated_weather_data/simulation_data" + str(paths) + "paths.csv")




