import os
import numpy as np 
import pandas as pd
from datetime import datetime, date, time, timedelta

from data_utils import *
from data_utils.cmaq import CMAQReader

baseline_cmaq_summation_species = {'FSPMC': ['ASO4J', 'ASO4I', 'ANO3J', 'ANO3I', 'ANH4J', 'ANH4I', 'AXYL1J', 'AALKJ', 'AXYL2J', 'AXYL3J', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ATRP1J', 'ATRP2J', 'AISO1J', 'AISO2J', 'ASQTJ', 'AORGCJ', 'AORGPAJ', 'AORGPAI', 'AECJ', 'AECI', 'A25J', 'A25I', 'ANAJ', 'ANAI', 'ACLJ', 'AISO3J', 'AOLGAJ', 'AOLGBJ']}
baseline_cmaq_species = ['O3']

if __name__ == '__main__':
    reader = CMAQReader()

    source_stations = set(load_dict(f'{data_dir}/source_loc.pkl').keys())
    train_target_stations = set(load_dict(f'{data_dir}/train_target_loc.pkl').keys())
    test_target_stations = set(load_dict(f'{data_dir}/test_target_loc.pkl').keys())

    test_target_stations = list(test_target_stations - train_target_stations - source_stations)

    test_target_match = load_dict(f'{data_dir}/test_target_cmaq_match.pkl')
    test_target_locs = np.array([test_target_match[st] for st in test_target_stations])
    test_first_date, test_last_date = test_first_dt.date(), test_last_dt.date()
    cur_date = test_first_date + timedelta(days = history_days)
    
    cmaq_pred = []
    while cur_date <= test_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), test_target_locs, baseline_cmaq_species, baseline_cmaq_summation_species) 
        cmaq_pred.append(np.stack([dic[st].T for st in target_species], axis = -2))
        cur_date += timedelta(days = 1)
    cmaq_pred = np.stack(cmaq_pred)

    os.makedirs('./baselines/results', exist_ok= True)
    save_dict('./baselines/results/cmaq_pred.pkl', {st: cmaq_pred[:, i] for i, st in enumerate(test_target_stations)})
