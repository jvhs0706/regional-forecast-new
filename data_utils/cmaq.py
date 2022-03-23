import numpy as np 
from netCDF4 import Dataset
import os
import pandas as pd
from datetime import datetime, date, time, timedelta

from . import *

class CMAQReader:
    def __init__(self, cmaq_dir = '/home/dataop/data/nmodel/cmaq_fc', cmaq_version = 'CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC',
        begin_hour = 12, resolution = 3):
        self.cmaq_dir = cmaq_dir 
        self.cmaq_version = cmaq_version 
        self.begin_hour = begin_hour 
        self.resolution = resolution 
    
    def __call__(self, folder_date: date, first_dt: datetime, last_dt: datetime, locs: np.array = None, species: list = cmaq_species, summation_species: dict = cmaq_summation_species):
        '''
        Use UTC time zone.
        locs: np.array with shape (num_loc_0, num_loc_1, ..., num_loc_d-1, 2) 
        '''
        regional_data = {sp: [] for sp in species + list(summation_species.keys())}
        if locs is not None:
            loc_ind_0, loc_ind_1 = locs[..., 0].flatten(), locs[..., 1].flatten()
        else:
            loc_ind_0, loc_ind_1 = slice(None), slice(None)

        directory = f'{self.cmaq_dir}/{folder_date.year}/{folder_date.year}{folder_date.month:02}/{folder_date.year}{folder_date.month:02}{folder_date.day:02}{self.begin_hour:02}/{self.resolution}km'
        ds_first_date, ds_last_date = (first_dt - timedelta(hours = self.begin_hour)).date(), (last_dt - timedelta(hours = self.begin_hour)).date()

        for ds_date in pd.date_range(ds_first_date, ds_last_date):
            fn = f'{self.cmaq_version}.{ds_date.year}{(ds_date.date() - date(ds_date.year, 1, 1)).days + 1:03}'
            ds = Dataset(f'{directory}/{fn}')
            idx = slice((first_dt.hour - self.begin_hour)%24 if ds_date.date() == ds_first_date else 0, 
                (last_dt.hour - self.begin_hour)%24 + 1 if ds_date.date() == ds_last_date else 24)
            if locs is not None:
                for sp in species:
                    regional_data[sp].append(ds[sp][idx, :].data.mean(axis = 1)[:, loc_ind_0, loc_ind_1].reshape(-1, *locs.shape[:-1]) * 1000)
                for sp, components in summation_species.items():
                    regional_data[sp].append(np.sum([ds[c][idx, :].data.mean(axis = 1)[:, loc_ind_0, loc_ind_1].reshape(-1, *locs.shape[:-1]) for c in components], axis = 0))
            else:
                for sp in species:
                    regional_data[sp].append(ds[sp][idx, :].data.mean(axis = 1) * 1000)
                for sp, components in summation_species.items():
                    regional_data[sp].append(np.sum([ds[c][idx, :].data.mean(axis = 1) for c in components], axis = 0))
            ds.close()

        regional_data = {sp: np.concatenate(arr) for sp, arr in regional_data.items()}
        return regional_data

            
if __name__ == '__main__':
    train_target_loc = load_dict(f'{data_dir}/train_target_loc.pkl')
    test_target_loc = load_dict(f'{data_dir}/test_target_loc.pkl')
    cmaq_latlon = np.load(f'{data_dir}/CMAQ.GRIDCRO2D.3km.npz')
    cmaq_latlon = np.stack([cmaq_latlon['LAT'], cmaq_latlon['LON']], axis = -1)
    dist_train_target = batch_dist(cmaq_latlon, np.array(list(train_target_loc.values())))
    dist_test_target = batch_dist(cmaq_latlon, np.array(list(test_target_loc.values())))
    train_target_match = np.stack(np.unravel_index(dist_train_target.reshape(-1, len(train_target_loc)).argmin(axis = 0), shape = cmaq_latlon.shape[:-1]), axis = 1)
    test_target_match = np.stack(np.unravel_index(dist_test_target.reshape(-1, len(test_target_loc)).argmin(axis = 0), shape = cmaq_latlon.shape[:-1]), axis = 1)
    save_dict(f'{data_dir}/train_target_cmaq_match.pkl', {st: tuple(arr) for st, arr in zip(train_target_loc.keys(), train_target_match)})
    save_dict(f'{data_dir}/test_target_cmaq_match.pkl', {st: tuple(arr) for st, arr in zip(test_target_loc.keys(), test_target_match)})

    reader = CMAQReader()
    train_target_match = load_dict(f'{data_dir}/train_target_cmaq_match.pkl')
    train_target_locs = np.array(list(train_target_match.values()))
    train_first_date, train_last_date = train_first_dt.date(), train_last_dt.date()
    cur_date = train_first_date + timedelta(days = history_days)

    train_target_cmaq_data = []
    while cur_date <= train_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), train_target_locs) 
        train_target_cmaq_data.append(np.stack([arr.T for arr in dic.values()], axis = -2))
        cur_date += timedelta(days = 1)
    train_target_cmaq_data = np.stack(train_target_cmaq_data)
    np.save(f'{data_dir}/train_target_cmaq_data.npy', train_target_cmaq_data)

    cmaq_mean, cmaq_std = np.mean(train_target_cmaq_data, axis = (0, 1, 3)), np.std(train_target_cmaq_data, axis = (0, 1, 3))
    np.savez(f'{data_dir}/cmaq_normalizing.npz', mean = cmaq_mean, std = cmaq_std)

    test_target_match = load_dict(f'{data_dir}/test_target_cmaq_match.pkl')
    test_target_locs = np.array(list(test_target_match.values()))
    test_first_date, test_last_date = test_first_dt.date(), test_last_dt.date()
    cur_date = test_first_date + timedelta(days = history_days)
    
    test_target_cmaq_data = []
    while cur_date <= test_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), test_target_locs) 
        test_target_cmaq_data.append(np.stack([arr.T for arr in dic.values()], axis = -2))
        cur_date += timedelta(days = 1)
    test_target_cmaq_data = np.stack(test_target_cmaq_data)
    np.save(f'{data_dir}/test_target_cmaq_data.npy', test_target_cmaq_data)