import os
import pandas as pd 
import numpy as np 

from datetime import datetime, time 
import argparse 

from .interpolation import *
from . import *

source_threshold = 0.05
target_threshold = 0.1

timedelta_to_hours = lambda td: td.days * 24 + td.seconds // 3600
dt_to_idx = lambda dt: timedelta_to_hours(dt - datetime.combine(obs_first_date, time(0))) 

def _print_dic(dic, N = 5):
    print(len(dic))
    for (st, df), _ in zip(dic.items(), range(N)):
        print(st, df.shape)
        print(df.head())

def remove_invalid(x):
    try:
        return float(x) if x > -999.0 else np.NaN
    except:
        return np.NaN

def read_csv(fn: str, index_row: int, ndays: int):
    loc = pd.read_csv(fn, skiprows = lambda x: x not in range(index_row, index_row + 3), index_col = 0).applymap(remove_invalid)
    df = pd.read_csv(fn, skiprows = lambda x: x != index_row and x not in range(index_row + 4, index_row + 4 + 24 * ndays), index_col = 0).applymap(remove_invalid)
    df['time'] = pd.to_datetime(df.index.values, format = '%Y/%m/%d %H:%M:%S')
    df.set_index('time', inplace = True)
    return loc, df

timedelta_to_hours = lambda td: td.days * 24 + td.seconds // 3600
dt_to_idx = lambda dt: timedelta_to_hours(dt - datetime.combine(obs_first_date, time()))

class ObsReader:
    def __init__(self, data_dic, target_stations = None):
        if target_stations is not None:
            self.data_dic = {st: data_dic[st] for st in target_stations}
        else:
            self.data_dic = data_dic
    def __call__(self, first_dt: datetime, last_dt: datetime):
        return {st: df.loc[first_dt:last_dt] for st, df in self.data_dic.items()}

if __name__ == '__main__':
    obs_first_date_str,obs_last_date_str = obs_first_date.strftime('%Y%m%d'),obs_last_date.strftime('%Y%m%d')
    
    # load the target data
    normalizing = {}

    train_target_stations, test_target_stations = [], []
    train_target_data, test_target_data = {}, {}
    
    for target_sp in target_species:
        print(f'Processing {target_sp} (target)...')
        fn = f'{data_dir}/AQ_{target_sp}-'+obs_first_date.strftime('%Y%m%d')+'-'+obs_last_date.strftime('%Y%m%d')+'.csv'
        loc, df = read_csv(fn, index_row=5, ndays=(obs_last_date - obs_first_date).days + 1)
        
        train_mask, test_mask = np.isnan(df[train_first_dt:train_last_dt].values).mean(axis = 0) < target_threshold, \
            np.isnan(df[test_first_dt:test_last_dt].values).mean(axis = 0) < target_threshold
        train_target_stations.append(set(loc.columns[train_mask])), test_target_stations.append(set(loc.columns[test_mask]))
        train_target_data[target_sp], test_target_data[target_sp] = df.loc[train_first_dt:train_last_dt, list(train_target_stations[-1])],\
            df.loc[test_first_dt:test_last_dt, list(test_target_stations[-1])]

    train_target_stations, test_target_stations = set.intersection(*train_target_stations), set.intersection(*test_target_stations)
    train_target_loc, test_target_loc = {st: tuple(loc[st]) for st in train_target_stations}, {st: tuple(loc[st]) for st in test_target_stations}
    train_target_data, test_target_data = {st: pd.DataFrame.from_dict({target_sp: train_target_data[target_sp][st] for target_sp in target_species}) for st in train_target_stations},\
        {st: pd.DataFrame.from_dict({target_sp: test_target_data[target_sp][st] for target_sp in target_species}) for st in test_target_stations}

    print(f'Number of training target stations: {len(train_target_stations)}.')
    print(f'Number of testing target stations: {len(test_target_stations)}.')
        
    save_dict(data_dir + '/train_target_loc.pkl', train_target_loc)
    save_dict(data_dir + '/train_target_data.pkl', train_target_data)
    save_dict(data_dir + '/test_target_loc.pkl', test_target_loc)
    save_dict(data_dir + '/test_target_data.pkl', test_target_data)

    # load the source data of the target species
    source_stations = []
    source_loc = {}
    train_source_data, test_source_data = {}, {}

    for target_sp in target_species:
        print(f'Processing {target_sp} (source)...')
        fn = f'{data_dir}/AQ_{target_sp}-'+obs_first_date.strftime('%Y%m%d')+'-'+obs_last_date.strftime('%Y%m%d')+'.csv'
        loc, df = read_csv(fn, index_row=5, ndays=(obs_last_date - obs_first_date).days + 1)

        normalizing[target_sp] = (np.nanmean(df[train_first_dt:train_last_dt].values), np.nanstd(df[train_first_dt:train_last_dt].values))
        
        mask = np.logical_and(np.isnan(df[train_first_dt:train_last_dt].values).mean(axis = 0) < source_threshold, np.isnan(df[test_first_dt:test_last_dt].values).mean(axis = 0) < source_threshold)
        source_stations.append(set(loc.columns[mask]))
        train_source_data[target_sp], test_source_data[target_sp] = df.loc[train_first_dt:train_last_dt, list(source_stations[-1])], \
            df.loc[test_first_dt:test_last_dt, list(source_stations[-1])]

    source_stations = set.intersection(*source_stations)
    source_loc = {st: tuple(loc[st]) for st in source_stations}
    train_source_data, test_source_data = {st: pd.DataFrame.from_dict({target_sp: train_source_data[target_sp][st] for target_sp in target_species}) for st in source_stations},\
        {st: pd.DataFrame.from_dict({target_sp: test_source_data[target_sp][st] for target_sp in target_species}) for st in source_stations}
    
    # load the source data of the non-target species
    for sp in AQ_species:
        if sp not in target_species:
            print(f'Processing {sp}...')
            fn = f'{data_dir}/AQ_{sp}-'+obs_first_date.strftime('%Y%m%d')+'-'+obs_last_date.strftime('%Y%m%d')+'.csv'
            loc, df = read_csv(fn, index_row=5, ndays=(obs_last_date - obs_first_date).days + 1)
            normalizing[sp] = (np.nanmean(df[train_first_dt:train_last_dt].values), np.nanstd(df[train_first_dt:train_last_dt].values))

            mask = np.logical_and(np.isnan(df[train_first_dt:train_last_dt].values).mean(axis = 0) < source_threshold, np.isnan(df[test_first_dt:test_last_dt].values).mean(axis = 0) < source_threshold)
            for st in set.intersection(set(loc.columns[mask]), source_stations):
                train_source_data[st][sp], test_source_data[st][sp] = df.loc[train_first_dt:train_last_dt, st],\
                    df.loc[test_first_dt:test_last_dt, st]

    print(f'Number of source stations: {len (source_stations)}.')
    print(f'Number of non-source non-training testing target stations: {len(test_target_stations - source_stations - train_target_stations)}.')

    # load the weather data
    for sp in A_species:
        print(f'Processing {sp}...')
        fn = f'{data_dir}/A_{sp}-'+obs_first_date.strftime('%Y%m%d')+'-'+obs_last_date.strftime('%Y%m%d')+'.csv'
        if sp == 'WIND':
            loc, speed = read_csv(fn, index_row=5, ndays=(obs_last_date - obs_first_date).days + 1)
            loc, direction = read_csv(fn, index_row=5 + ((obs_last_date - obs_first_date).days + 1) * 24 + 6, ndays=(obs_last_date - obs_first_date).days + 1)
            direction = np.deg2rad(direction)
            wind = [speed * np.cos(direction), speed * np.sin(direction)]
            for i, df in enumerate(wind):
                normalizing[f'WIND.{i+1}'] = (np.nanmean(df[train_first_dt:train_last_dt].values), np.nanstd(df[train_first_dt:train_last_dt].values))

            interpolations = [InverseDistanceWeighted(lat_lon = loc.values.T, values = df.values.T) for df in wind]
            wind_interpolated = [interp(np.array(list(source_loc.values()))) for interp in interpolations]
            for i, arr in enumerate(wind_interpolated):
                for st, z in zip(source_loc, arr):
                    train_source_data[st][f'WIND.{i+1}'] = z[dt_to_idx(train_first_dt):dt_to_idx(train_last_dt) + 1]
                    test_source_data[st][f'WIND.{i+1}'] = z[dt_to_idx(test_first_dt):dt_to_idx(test_last_dt) + 1]

        else:
            loc, df = read_csv(fn, index_row=5, ndays=(obs_last_date - obs_first_date).days + 1)
            normalizing[sp] = (np.nanmean(df[train_first_dt:train_last_dt].values), np.nanstd(df[train_first_dt:train_last_dt].values))

            interpolation = InverseDistanceWeighted(lat_lon = loc.values.T, values = df.values.T)
            arr = interpolation(np.array(list(source_loc.values())))
            for st, z in zip(source_loc, arr):
                train_source_data[st][sp] = z[dt_to_idx(train_first_dt):dt_to_idx(train_last_dt) + 1]
                test_source_data[st][sp] = z[dt_to_idx(test_first_dt):dt_to_idx(test_last_dt) + 1]

    save_dict(data_dir + '/source_loc.pkl', source_loc)
    save_dict(data_dir + '/train_source_data.pkl', train_source_data)
    save_dict(data_dir + '/test_source_data.pkl', test_source_data)
    save_dict(data_dir + '/obs_normalizing.pkl', normalizing)