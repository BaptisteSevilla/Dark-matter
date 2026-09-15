import numpy as np

from astropy.cosmology import Planck18 as cosmo


R = np.logspace(-8, 2, 500) # [1e-8, 1e2] Mpc 
# Some plots want 1e-6 but we need to be lower to stabilize the algo

DM_PROFILE = 'NFW' # 'NFW', 'Einasto', 'Burkert', 'PIEMD', 'Zhao', 'Interpolated'

BARYON_PROFILE = 'Schneider2025' # 'Disk', 'Hernquist', 'Nuker', 'Schneider2015', 'Schneider2019',
                                 #'Schneider2025','Interpolated'

CONTRACTION = 'Ando' # 'Blumenthal', 'Ando', 'Schneider2015', 'Schneider2019', 'Schneider2025'

# Schneider et al. (2012) Parameters
HALO_MASS = 1e14 # Total mass of the halo in M_sun / h 
CONCENTRATION = 3.2 # Concentration parameter (c)

ALPHA_CUSTOM = 2.0
BETA_CUSTOM = 3.0
GAMMA_CUSTOM = 2.0

SCHNEIDER2015_CASE = 'B' 

GAS = True
FORCE_TRUNCATED_NFW_FOR_SCHNEIDER = True