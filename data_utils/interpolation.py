import numpy as np 
# from pykrige.ok import OrdinaryKriging
# from scipy.interpolate import SmoothSphereBivariateSpline
from geopy.distance import geodesic

from . import *

class Interpolation:
    def __init__(self, lat_lon, values):
        N, _ = lat_lon.shape
        self.lat_lon = lat_lon # (N, 2)
        self.values = values # (N, *)

    def interpolate(self, target_lat_lon, z):
        raise NotImplementedError

    def __call__(self, target_lat_lon):
        return np.apply_along_axis(lambda z: self.interpolate(target_lat_lon.reshape(-1, 2), z), 0, self.values).reshape(*target_lat_lon.shape[:-1], *self.values.shape[1:])

class NearestNeighbor(Interpolation):
    def __init__(self, lat_lon, values):
        super().__init__(lat_lon, values)
        
    def __call__(self, target_lat_lon):
        dist = batch_dist(target_lat_lon.reshape(-1, 2), self.lat_lon) # (T, S)
        indices = np.argmin(dist, axis = 1)

        return self.values[indices].reshape(*target_lat_lon.shape[:-1], *self.values.shape[1:])


class InverseDistanceWeighted(Interpolation):
    def __init__(self, lat_lon, values):
        super().__init__(lat_lon, values)
        
    def __call__(self, target_lat_lon, epsilon = 1e-8):
        dist = np.maximum(batch_dist(target_lat_lon.reshape(-1, 2), self.lat_lon), epsilon) # (T, S)
        unnormalized_weights = 1/dist
        values_flattened = self.values.reshape(self.values.shape[0], -1)
        out = (unnormalized_weights @ np.nan_to_num(values_flattened)) / (unnormalized_weights @ ~np.isnan(values_flattened))
        return out.reshape(*target_lat_lon.shape[:-1], *self.values.shape[1:])

# class Kriging(Interpolation):
#     def __init__(self, lat_lon, values):
#         super().__init__(lat_lon, values)

#     def interpolate(self, target_lat_lon, z):
#         mask = np.isnan(z)
#         kriging_instance = OrdinaryKriging(x = self.lat_lon[~mask, 1], y = self.lat_lon[~mask, 0], z = z[~mask], variogram_model='spherical', coordinates_type = 'geographic')
#         out, _ = kriging_instance.execute('points', target_lat_lon[:, 1], target_lat_lon[:, 0])
#         return out

# class SphericalSpline(Interpolation):
#     @staticmethod
#     def to_polar_coords(lat_lon):
#         theta, phi = np.PI / 2 - np.deg2rad(lat_lon[:, 0]), np.deg2rad(lat_lon[:, 1])
    
#     def __init__(self, lat_lon, values):
#         super().__init__()
    
#     def interpolate(self, target_lat_lon, z):
#         return SmoothSphereBivariateSpline(*to_polar_coords(self.lat_lon), z)(*to_polar_coords(target_lat_lon))
        
        