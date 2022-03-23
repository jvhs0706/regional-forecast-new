from data_utils.wrf import *
from data_utils.utils import *

import os
import pandas as pd
from datetime import datetime, date, time, timedelta
import numpy as np 
from netCDF4 import Dataset

if __name__ == '__main__':
    source_loc = load_dict(f'{data_dir}/source_loc.pkl')
    wrf_latlon = np.load(f'{data_dir}/WRF.GRIDCRO2D.3km.npz')
    wrf_latlon = np.stack([wrf_latlon['LAT'], wrf_latlon['LON']], axis = -1)
    dist_source = batch_dist(wrf_latlon, np.array(list(source_loc.values())))
    
    source_match = np.stack(np.unravel_index(dist_source.reshape(-1, len(source_loc)).argmin(axis = 0), shape = wrf_latlon.shape[:-1]), axis = 1)
    save_dict(f'{data_dir}/source_wrf_match.pkl', {st: tuple(arr) for st, arr in zip(source_loc.keys(), source_match)})

    reader = WRFReader()

    source_match = load_dict(f'{data_dir}/source_wrf_match.pkl')
    source_locs = np.array(list(source_match.values()))
    
    train_first_date, train_last_date = train_first_dt.date(), train_last_dt.date()
    cur_date = train_first_date + timedelta(days = history_days)
    
    train_source_wrf_data = []
    while cur_date <= train_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), source_locs) 
        train_source_wrf_data.append(np.stack([arr.T for arr in dic.values()], axis = -2))
        cur_date += timedelta(days = 1)
    train_source_wrf_data = np.stack(train_source_wrf_data)
    np.save(f'{data_dir}/train_source_wrf_data.npy', train_source_wrf_data)

    test_first_date, test_last_date = test_first_dt.date(), test_last_dt.date()
    cur_date = test_first_date + timedelta(days = history_days)
    
    test_source_wrf_data = []
    while cur_date <= test_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), source_locs) 
        test_source_wrf_data.append(np.stack([arr.T for arr in dic.values()], axis = -2))
        cur_date += timedelta(days = 1)
    test_source_wrf_data = np.stack(test_source_wrf_data)
    np.save(f'{data_dir}/test_source_wrf_data.npy', test_source_wrf_data)

