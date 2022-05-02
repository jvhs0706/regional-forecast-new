import pickle as pk
from datetime import date, datetime, timedelta
import numpy as np
from geopy.distance import geodesic 

AQ_species = ['FSPMC', 'O3', 'NO2', 'SO2', 'RSPMC', 'CO']
A_species = ['TEMP', 'WIND', 'PRECIP', 'DEW_PT', 'PRE_SLP', 'RH_VT']
wrf_species = ['PSFC', 'U10', 'V10', 'T2', 'Q2']
cmaq_species = ['NO2', 'O3', 'SO2', 'CO']
cmaq_summation_species = {'FSPMC': ['ASO4J', 'ASO4I', 'ANO3J', 'ANO3I', 'ANH4J', 'ANH4I', 'AXYL1J', 'AALKJ', 'AXYL2J', 'AXYL3J', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ATRP1J', 'ATRP2J', 'AISO1J', 'AISO2J', 'ASQTJ', 'AORGCJ', 'AORGPAJ', 'AORGPAI', 'AECJ', 'AECI', 'A25J', 'A25I', 'ANAJ', 'ANAI', 'ACLJ', 'AISO3J', 'AOLGAJ', 'AOLGBJ']}
target_species = ['FSPMC', 'O3']

data_dir = 'data'

obs_first_date = date(2015,1,1)
obs_last_date = date(2021,12,31)

train_first_dt, train_last_dt = datetime(2015, 1, 1, 0), datetime(2020, 12, 31, 23)
test_first_dt, test_last_dt = datetime(2021, 1, 1, 0), datetime(2021, 12, 31, 23)
timezone = -1
history_days, horizon_days = 3, 2

EARTH_RADIUS = 6371.009

'''
Data I/O utilities
Loading and saving dictionaries.
'''

def save_dict(fn: str, dic: dict):
    with open(fn, 'wb') as f:
        pk.dump(dic, f)

def load_dict(fn: str):
    with open(fn, 'rb') as f:
        return pk.load(f)

'''
distance between a set of source locations and a set of target locations
'''
hav = lambda z: np.sin(z/2)**2
archav = lambda z: 2 * np.arcsin(np.sqrt(z))

def batch_dist(batch_0: np.array, batch_1: np.array):
    (lat_0, lon_0), (lat_1, lon_1) = np.deg2rad(batch_0).reshape(-1, 2).T, np.deg2rad(batch_1).reshape(-1, 2).T
    lat_abs_diff, lon_abs_diff = np.abs(lat_0[:, None] - lat_1[None, :]), np.abs(lon_0[:, None] - lon_1[None, :])
    out = archav(hav(lat_abs_diff) + (1-hav(lat_abs_diff)-hav((lat_0[:, None] + lat_1[None, :])))*hav(lon_abs_diff)) * EARTH_RADIUS

    if len(batch_0.shape) + len(batch_1.shape) > 2:
        return out.reshape(*batch_0.shape[:-1], *batch_1.shape[:-1])
    else:
        return out[0, 0]

'''
Total mean and variation:
E(Y) = E(E(Y|X))
Var(Y) = E(Var(Y|X)) + Var(E(Y|X))
'''
def total_mean_std(means: np.array, std: np.array):
    return means.mean(), np.sqrt(np.mean(std**2) + np.var(means))