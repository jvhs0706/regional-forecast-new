import numpy as np
import os

from data_utils import *

lat_range = np.linspace(21.6, 24.5)
lon_range = np.linspace(111.2, 115.6)

lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
grid = np.stack([lat_grid, lon_grid], axis = -1).reshape(-1, 2)