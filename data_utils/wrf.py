import numpy as np 
from netCDF4 import Dataset
import os
import pandas as pd
from datetime import datetime, date, time, timedelta

from . import *

class WRFReader:
    def __init__(self, wrf_dir = '/home/dataop/data/nmodel/wrf_fc', begin_hour = 12, pred_class = 'd03'):
        self.wrf_dir = wrf_dir
        self.begin_hour = begin_hour 
        self.pred_class = pred_class 
    
    def __call__(self, folder_date: date, first_dt: datetime, last_dt: datetime, locs: np.array = None, species: list = wrf_species):
        '''
        Use UTC time zone.
        locs: np.array with shape (num_loc_0, num_loc_1, ..., num_loc_d-1, 2) 
        '''
        regional_data = {sp: [] for sp in species}
        if locs is not None:
            loc_ind_0, loc_ind_1 = locs[..., 0].flatten(), locs[..., 1].flatten()

        directory = f'{self.wrf_dir}/{folder_date.year}/{folder_date.year}{folder_date.month:02}/{folder_date.year}{folder_date.month:02}{folder_date.day:02}{self.begin_hour:02}'
        ds_first_date, ds_last_date = (first_dt - timedelta(hours = self.begin_hour)).date(), (last_dt - timedelta(hours = self.begin_hour)).date()

        cur_dt = first_dt
        while cur_dt <= last_dt:
            cur_dt_str = cur_dt.strftime('%Y-%m-%d_%H:%M:%S')
            fn = f'wrfout_{self.pred_class}_{cur_dt_str}'
            ds = Dataset(f'{directory}/{fn}')
            for sp in species:
                if locs is not None:
                    regional_data[sp].append(np.squeeze(ds[sp])[loc_ind_0, loc_ind_1].reshape(*locs.shape[:-1]))
                else:
                    regional_data[sp].append(np.squeeze(ds[sp]))
            cur_dt += timedelta(hours = 1)
            ds.close()
        
        regional_data = {sp: np.stack(arr) for sp, arr in regional_data.items()}
        return regional_data

            
if __name__ == '__main__':
    train_target_loc = load_dict(f'{data_dir}/train_target_loc.pkl')
    test_target_loc = load_dict(f'{data_dir}/test_target_loc.pkl')
    wrf_latlon = np.load(f'{data_dir}/WRF.GRIDCRO2D.3km.npz')
    wrf_latlon = np.stack([wrf_latlon['LAT'], wrf_latlon['LON']], axis = -1)
    dist_train_target = batch_dist(wrf_latlon, np.array(list(train_target_loc.values())))
    dist_test_target = batch_dist(wrf_latlon, np.array(list(test_target_loc.values())))
    train_target_match = np.stack(np.unravel_index(dist_train_target.reshape(-1, len(train_target_loc)).argmin(axis = 0), shape = wrf_latlon.shape[:-1]), axis = 1)
    test_target_match = np.stack(np.unravel_index(dist_test_target.reshape(-1, len(test_target_loc)).argmin(axis = 0), shape = wrf_latlon.shape[:-1]), axis = 1)
    save_dict(f'{data_dir}/train_target_wrf_match.pkl', {st: tuple(arr) for st, arr in zip(train_target_loc.keys(), train_target_match)})
    save_dict(f'{data_dir}/test_target_wrf_match.pkl', {st: tuple(arr) for st, arr in zip(test_target_loc.keys(), test_target_match)})

    reader = WRFReader()
    
    train_target_match = load_dict(f'{data_dir}/train_target_wrf_match.pkl')
    train_target_locs = np.array(list(train_target_match.values()))
    train_first_date, train_last_date = train_first_dt.date(), train_last_dt.date()
    cur_date = train_first_date + timedelta(days = history_days)
    
    train_target_wrf_data = []
    while cur_date <= train_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), train_target_locs) 
        train_target_wrf_data.append(np.stack([arr.T for arr in dic.values()], axis = -2))
        cur_date += timedelta(days = 1)
    train_target_wrf_data = np.stack(train_target_wrf_data)
    np.save(f'{data_dir}/train_target_wrf_data.npy', train_target_wrf_data)

    wrf_mean, wrf_std = np.mean(train_target_wrf_data, axis = (0, 1, 3)), np.std(train_target_wrf_data, axis = (0, 1, 3))
    np.savez(f'{data_dir}/wrf_normalizing.npz', mean = wrf_mean, std = wrf_std)

    test_target_match = load_dict(f'{data_dir}/test_target_wrf_match.pkl')
    test_target_locs = np.array(list(test_target_match.values()))
    test_first_date, test_last_date = test_first_dt.date(), test_last_dt.date()
    cur_date = test_first_date + timedelta(days = history_days)
    
    test_target_wrf_data = []
    while cur_date <= test_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), test_target_locs) 
        test_target_wrf_data.append(np.stack([arr.T for arr in dic.values()], axis = -2))
        cur_date += timedelta(days = 1)
    test_target_wrf_data = np.stack(test_target_wrf_data)
    np.save(f'{data_dir}/test_target_wrf_data.npy', test_target_wrf_data)