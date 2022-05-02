import numpy as np
import os

from data_utils import *
from . import *

cmaq_latlon = np.load(f'{data_dir}/CMAQ.GRIDCRO2D.3km.npz')
cmaq_latlon = np.stack([cmaq_latlon['LAT'], cmaq_latlon['LON']], axis = -1)

cmaq_dist = batch_dist(cmaq_latlon, grid)
cmaq_match = np.stack(np.unravel_index(cmaq_dist.reshape(-1, len(grid)).argmin(axis = 0), shape = cmaq_latlon.shape[:-1]), axis = 1)

wrf_latlon = np.load(f'{data_dir}/WRF.GRIDCRO2D.3km.npz')
wrf_latlon = np.stack([wrf_latlon['LAT'], wrf_latlon['LON']], axis = -1)

wrf_dist = batch_dist(wrf_latlon, grid)
wrf_match = np.stack(np.unravel_index(wrf_dist.reshape(-1, len(grid)).argmin(axis = 0), shape = wrf_latlon.shape[:-1]), axis = 1)

os.makedirs('regional_forecast/data', exist_ok = True)
np.savez('regional_forecast/data/match.npz', WRF = wrf_match, CMAQ = cmaq_match)
np.save('regional_forecast/data/grid.npy', grid)