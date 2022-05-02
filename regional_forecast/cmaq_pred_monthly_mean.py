import argparse 
import numpy as np 
import sys

from datetime import datetime, date, time, timedelta

from data_utils import *
from baselines.lstm_fc.cmaq import *
from data_utils.cmaq import CMAQReader

if __name__ == '__main__':
    year, month = int(sys.argv[1]), int(sys.argv[2])
    begin_day = date(year, month, 1)
    end_day = date(year, month + 1, 1)

    match = np.load('regional_forecast/data/match.npz')
    cmaq_match = match['CMAQ']
    reader = CMAQReader()
    
    results = {}
    for i in range(2):
        monthly_arr = []
        for j in range((end_day - begin_day).days):
            pred_date = begin_day + timedelta(days = j)
            folder_date = pred_date - timedelta(days = 2 + i)
            first_dt = datetime.combine(pred_date, time(0)) + timedelta(hours = -timezone)
            last_dt = datetime.combine(pred_date, time(0)) + timedelta(hours = 24-1-timezone) 
            dic = reader(folder_date, first_dt, last_dt, cmaq_match, cmaq_species, baseline_cmaq_summation_species)
            monthly_arr.append(np.stack([dic[sp].T for sp in target_species], axis = -2))

        results[f'day{i}'] = np.mean(monthly_arr, axis = 0)

    os.makedirs('regional_forecast/data/cmaq-pred', exist_ok=True)
    np.savez(f'regional_forecast/data/cmaq-pred/{year}.{month}.npz', **results)
