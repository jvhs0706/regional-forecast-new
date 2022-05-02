import argparse 
import numpy as np 
import sys

from datetime import datetime, date, time, timedelta

from data_utils import *
from baselines.lstm_fc.cmaq import *
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
    cmaq_match = match['CMAQ']
    reader = CMAQReader()
    
    results = {}
    for i in range(2):    
        folder_date = pred_date - timedelta(days = 2 + i)
        first_dt = datetime.combine(pred_date, time(0)) + timedelta(hours = -timezone)
        last_dt = datetime.combine(pred_date, time(0)) + timedelta(hours = 24-1-timezone) 
        dic = reader(folder_date, first_dt, last_dt, cmaq_match, cmaq_species, baseline_cmaq_summation_species)

        results[f'day{i}'] = np.stack([dic[sp].T for sp in target_species], axis = -2)

    os.makedirs('regional_forecast/data/cmaq-pred', exist_ok=True)
    np.savez(f'regional_forecast/data/cmaq-pred/{pred_date}.npz', **results)
