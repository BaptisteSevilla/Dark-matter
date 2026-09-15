import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class InterpolatedProfile:
    def __init__(self, filepath, x_col=0, y_col=1, sep=';', decimal=',', log_space=True):
        self.df = pd.read_csv(filepath, sep=sep, decimal=decimal, header=None)
        self.df = self.df.sort_values(by=x_col)
        
        self.x = self.df.iloc[:, x_col].values
        self.y = self.df.iloc[:, y_col].values
        self.log_space = log_space
        
        self.r_min = np.min(self.x[self.x > 0]) if self.log_space else np.min(self.x)
        self.r_max = np.max(self.x)
        
        if self.log_space:
            valid_mask = (self.x > 0) & (self.y > 0)
            log_x = np.log10(self.x[valid_mask])
            log_y = np.log10(self.y[valid_mask])
            self.interp_func = interp1d(log_x, log_y, kind='linear', bounds_error=False, fill_value=np.nan)
        else:
            self.interp_func = interp1d(self.x, self.y, kind='linear', bounds_error=False, fill_value=np.nan)
            
    def __call__(self, r):
        r = np.asarray(r)
        result = np.full_like(r, np.nan, dtype=float)
        
        in_bounds = (r >= self.r_min) & (r <= self.r_max)
        
        if not np.any(in_bounds):
            return result
            
        if self.log_space:
            result[in_bounds] = 10**(self.interp_func(np.log10(r[in_bounds])))
        else:
            result[in_bounds] = self.interp_func(r[in_bounds])
            
        return result
        
    def get_bounds(self):
        return self.r_min, self.r_max