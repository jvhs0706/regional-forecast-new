import os
from datetime import date

import numpy as np 
from netCDF4 import Dataset

from . import *

if __name__ == '__main__':
    first_date_str, last_date_str = obs_first_date.strftime('%Y%m%d'), obs_last_date.strftime('%Y%m%d')
    for fn in os.listdir(data_dir):
        if fn[-4:] == '.csv':
            with open(f'{data_dir}/{fn}', 'r') as f:
                title_line = f.readline()
                print(f'Verifying: {title_line[12:-2]}')
                info_line = f.readline()
                assert f'Time(UTC{timezone:+}): {first_date_str}-{last_date_str}' == info_line[1:-2], f'Wrong signature in {fn}!'
    
    print('All data signature tests passed!')

    
    