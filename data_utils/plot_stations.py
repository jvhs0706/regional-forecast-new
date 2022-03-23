import matplotlib.pyplot as plt
import numpy as np

from . import *

def get_border(fn: str, lat_range = (-np.inf, np.inf), lon_range = (-np.inf, np.inf)):
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    with open(fn, 'r') as f:
        txt = f.read().split('\n')
    out, temp = [], {'lon':[], 'lat':[]}
    for t in txt:
        if len(t) == 0:
            out.append((np.array(temp['lon']), np.array(temp['lat'])))
            temp = {'lon':[], 'lat':[]}
        else:
            lon, lat = t.split('     ')
            if lon_min <= float(lon) <= lon_max and lat_min <= float(lat) <= lat_max:
                temp['lon'].append(float(lon))
                temp['lat'].append(float(lat))
            else:
                out += [(np.array(temp['lon']), np.array(temp['lat']))]
                temp = {'lon':[], 'lat':[]}

    return out

def get_loc(dic: dict):
    lats, lons = [], []
    for st, loc in dic.items():
        lat, lon = loc
        lats.append(lat), lons.append(lon)
    return np.array(lons), np.array(lats) 

if __name__ == '__main__':
    fig, ax = plt.subplots(constrained_layout = True)

    source_loc = load_dict(f'{data_dir}/source_loc.pkl')
    lons, lats = get_loc(source_loc)
    ax.plot(lons, lats, 'or', markersize = 2, label = 'Source stations')
    

    train_target_loc = load_dict(f'{data_dir}/train_target_loc.pkl')
    lons, lats = get_loc({st: loc for st, loc in train_target_loc.items() if st not in source_loc})
    ax.plot(lons, lats, 'ob', markersize = 2, label = 'Non-source training target stations')

    test_target_loc = load_dict(f'{data_dir}/test_target_loc.pkl')
    lons, lats = get_loc({st: loc for st, loc in test_target_loc.items() if st not in source_loc and st not in train_target_loc})
    ax.plot(lons, lats, 'og', markersize = 2, label = 'Non-source non-training testing target stations')

    ax.legend()
    
    borders = get_border(f'{data_dir}/pearl_delta.txt')
    for section in borders:
        ax.plot(*section, 'k', linewidth=1)

    fig.savefig('station_locations.png')