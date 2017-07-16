import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')

directory = os.getcwd()


pepco_data = pd.read_csv(directory + '/pepco_price.csv')
pepco_data['rt_time'] = list(map(str, pepco_data['rt_time']))
pepco_data['rt_time'] = [x if x!='2400' else '000' for x in pepco_data['rt_time']]


hourMins = [hourMin[:-2] + ':' + hourMin[-2:] for hourMin in pepco_data['rt_time']]
pepco_data['datetime'] = [date + ':' + hourMin for date, hourMin in zip(pepco_data['rt_date'], hourMins)]


pepco_data['datetime'] = list(map(lambda datetime: time.strptime(str(datetime), "%m/%d/%Y:%H:%M"), pepco_data['datetime']))
pepco_data['datetime'] = [time.strftime('%Y-%m-%dT%H:%M:%SZ', x) for x in pepco_data['datetime']]
pepco_data['rt_date'] = list(map(lambda datetime: datetime[:10], pepco_data['datetime']))
pepco_data.sort_values(['datetime'], ascending=[1], inplace=True)


minPrice = min([x for x  in pepco_data['lmp'] if x > 0] )
pepco_data['lmp'] = [x if x > 0 else minPrice for x in list(pepco_data['lmp'])]


pepco_data.to_csv(directory + '/pepco_price_cleaned.csv')