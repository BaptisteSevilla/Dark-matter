import numpy as np

from astropy.cosmology import Planck18 as cosmo


R = np.logspace(np.log10(0.02), np.log10(0.21), 500) * 0.70


DM_PROFILE = 'Zhao' # 'NFW', 'Einasto', 'Burkert', 'PIEMD', 'Zhao', 'Interpolated'

BARYON_PROFILE = 'PIEMD' 

CONTRACTION = 'Ando' # 'Blumenthal', 'Ando', 'Schneider2015', 'Schneider2019', 'Schneider2025'


HALO_MASS = 1.8e15 * 0.7# Total mass of the halo in M_sun / h 

CONCENTRATION = 3.1 # Concentration parameter (c)

ALPHA_CUSTOM = 1.0
BETA_CUSTOM = 3.0
GAMMA_CUSTOM = 1.09

SCHNEIDER2015_CASE = 'B' 

GAS = True
FORCE_TRUNCATED_NFW_FOR_SCHNEIDER = True