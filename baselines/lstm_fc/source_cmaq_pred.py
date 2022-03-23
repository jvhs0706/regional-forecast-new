from data_utils.cmaq import *
from data_utils.utils import *

import os
import pandas as pd
from datetime import datetime, date, time, timedelta
import numpy as np 
from netCDF4 import Dataset

baseline_cmaq_summation_species = {'FSPMC': ['ASO4J', 'ASO4I', 'ANO3J', 'ANO3I', 'ANH4J', 'ANH4I', 'AXYL1J', 'AALKJ', 'AXYL2J', 'AXYL3J', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ATRP1J', 'ATRP2J', 'AISO1J', 'AISO2J', 'ASQTJ', 'AORGCJ', 'AORGPAJ', 'AORGPAI', 'AECJ', 'AECI', 'A25J', 'A25I', 'ANAJ', 'ANAI', 'ACLJ', 'AISO3J', 'AOLGAJ', 'AOLGBJ']}
baseline_cmaq_species = ['O3']

if __name__ == '__main__':
    reader = CMAQReader()

    source_match = load_dict(f'{data_dir}/source_cmaq_match.pkl')
    source_locs = np.array(list(source_match.values()))
    source_stations = list(source_match.keys())
    
    test_first_date, test_last_date = test_first_dt.date(), test_last_dt.date()
    cur_date = test_first_date + timedelta(days = history_days)
    
    source_cmaq_pred = []
    while cur_date <= test_last_date - timedelta(days=horizon_days-1): 
        dic = reader(cur_date - timedelta(days = 2), datetime.combine(cur_date, time(0)) + timedelta(hours = -timezone), datetime.combine(cur_date, time(0)) + timedelta(hours = 24*horizon_days-1-timezone), source_locs, baseline_cmaq_species, baseline_cmaq_summation_species) 
        source_cmaq_pred.append(np.stack([dic[st].T for st in target_species], axis = -2))
        cur_date += timedelta(days = 1)
    source_cmaq_pred = np.stack(source_cmaq_pred)

    os.makedirs('./baselines/results', exist_ok= True)
    save_dict(f'./baselines/results/source_cmaq_pred.pkl', {st: source_cmaq_pred[:, i] for i, st in enumerate(source_stations)})

