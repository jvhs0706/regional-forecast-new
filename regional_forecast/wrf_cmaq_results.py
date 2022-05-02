import argparse 
import numpy as np 
import sys

from datetime import datetime, date, time, timedelta

from data_utils import *
from baselines.lstm_fc.cmaq import *

from data_utils.wrf import WRFReader
from data_utils.cmaq import CMAQReader

if __name__ == '__main__':
    if len(sys.argv) == 3:
        year, day_of_year = int(sys.argv[1]), int(sys.argv[2])
        pred_date = date(year, 1, 1) + timedelta(days = day_of_year - 1)
        assert pred_date.year == year
    else:
        assert len(sys.argv) == 4
        pred_date = date(*[int(a) for a in sys.argv[1:]])

    match = np.load('regional_forecast/data/match.npz')
    wrf_match, cmaq_match = match['WRF'], match['CMAQ']

    wrf_reader = WRFReader()
    cmaq_reader = CMAQReader()
    
    folder_date = pred_date - timedelta(days = 2)
    first_dt = datetime.combine(pred_date, time(0)) + timedelta(hours = -timezone)
    last_dt = datetime.combine(pred_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone) 

    dic = {**wrf_reader(folder_date, first_dt, last_dt, wrf_match), **cmaq_reader(folder_date, first_dt, last_dt, cmaq_match)}
    arr = np.stack([arr.T for arr in dic.values()], axis = -2)
    
    os.makedirs('regional_forecast/data/wrf-cmaq-results', exist_ok=True)
    np.save(f'regional_forecast/data/wrf-cmaq-results/{pred_date}.npy', arr)
