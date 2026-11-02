import numpy as np
import config

from astropy.cosmology import Planck18 as cosmo

# =============================================================================
# GENERAL CONSTANTS
# =============================================================================

TOLERANCE = 1e-9
MAX_EXTENSION_FACTOR = 1.35 #  >= 1 / (1 -F)  
NUMERICAL_SAFETY = 1e-10

Z = 0.28
H = 0.70

G = 4.30091e-9 

CONVERSION_FACTOR = 1e18 / H**2 #Conversion Ando to Schneider. h in M and h^3 in r^3

A_CONTRACT_RADIUS = 1e-10 # Intitial shift of r_i for iterative process

# =============================================================================
# COSMOLOGICAL CONSTANTS (Planck18)
# =============================================================================

#RHO_C = cosmo.critical_density(0).to("M_sun / Mpc3").value / H**2
RHO_C = cosmo.critical_density(Z).to("M_sun / Mpc3").value / H**2


OMEGA_M = 0.3  # Schneider et al. (2012)
OMEGA_B = 0.0
OMEGA_L = 0.7

RHO_M0 = RHO_C * OMEGA_M
RHO_M = RHO_C * OMEGA_M

F_B = OMEGA_B / OMEGA_M

RHO_BG = OMEGA_M * RHO_C 

# =============================================================================
# VIRIAL HALO PROPERTIES
# =============================================================================

# Bryan & Norman (1998)
# DELTA_VIR = (18.0 * np.pi**2 + 82.0 * (OMEGA_M - 1.0) - 39.0 * (OMEGA_M - 1.0)**2)
DELTA_VIR = 200
# Virial radius
R_VIR = (config.HALO_MASS / ((4.0 / 3.0) * np.pi * DELTA_VIR * RHO_C)) ** (1.0 / 3.0)

# =============================================================================
# GENERIC BARYONIC PARAMETERS 
# =============================================================================

# Generic Baryonic Parameters (Hernquist / Disk)
#  Milky Way size from Internet to be checked

R_B = 3.0e-3 * H  # Mpc/h

M_B = F_B * config.HALO_MASS  # Total baryonic mass


# =============================================================================
# EINASTO PROFILE (Ando 2012)
# =============================================================================

# Einasto parameters from Ando (2012)
N_EINASTO = 3.87
R_E_KPC = 1247.0
ALPHA_EINASTO = 1 / N_EINASTO


# =============================================================================
# ADIABATIC CONTRACTION MODEL (Ando / Gnedin et al. 2011)
# =============================================================================

A0 = 1.6  # Canonical value
W = 0.8   # Canonical value

# =============================================================================
# ANGULAR DISTANCES AND CONVERSION FACTORS
# =============================================================================

D_A_MPC = 13.7 / H
D_A_KPC = D_A_MPC * 1000.0

R_VIR_MPC = 1.452

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

ARCSEC_TO_MPC = D_A_MPC * ARCSEC_TO_RAD
ARCSEC_TO_KPC = D_A_KPC * ARCSEC_TO_RAD

ARCSEC_TO_RVIR = ARCSEC_TO_MPC / R_VIR_MPC

ESPSILON_BRIGHTNESS_MIN = 1e-5 # min = r * 1e-5
ESPSILON_BRIGHTNESS_MAX = 1e-8

ABEL_EPSREL = 1e-3  
ABEL_EPSREL_MIN = 1e-3 
ABEL_EPSREL_MAX = 1e-4
ABEL_LIMIT = 200
ABEL_EPSABS = 1-8

# =============================================================================
# FORNAX PARAMETERS
# =============================================================================

ALPHA_FORNAX = 1.58
BETA_FORNAX = 1.63
GAMMA_FORNAX = 0.09

THETA_B_FORNAX = 3.17
I_B_FORNAX = 17.23

M_L_RATIO = 10.0  # Mass-to-Light Ratio [M_sun/L_sun]
M_SUN_V = 4.83

R_B_KPC = THETA_B_FORNAX * ARCSEC_TO_KPC  # Break radius in kpc

I_B_LIN = (10.0**((M_SUN_V + 21.572 - I_B_FORNAX) / 2.5) * 1e6)  # L_sun/kpc^2


# =============================================================================
# GAS PROFILE (X-RAY BETA MODEL)
# =============================================================================

# Gas Profile

R_C1_ARCSEC = 3.93
BETA1_GAS = 0.54
NE0_1 = 0.3  # cm^-3

# Component 2 (NGC 1399 galactic halo)
R_C2_ARCSEC = 212.5
BETA2_GAS = 0.41
NE0_2 = 0.002  # cm^-3


# =============================================================================
# CONVERSION TO MPC/h UNITS
# =============================================================================

KPC_TO_MPCH = H / 1e3  # Conversion factor: kpc → Mpc/h

R_B_MPCH = R_B_KPC * KPC_TO_MPCH
R_C1_MPCH = (R_C1_ARCSEC * ARCSEC_TO_KPC) * KPC_TO_MPCH
R_C2_MPCH = (R_C2_ARCSEC * ARCSEC_TO_KPC) * KPC_TO_MPCH

I_B_LIN_MPCH = I_B_LIN * (1e3 / H)**2


# =============================================================================
# GAS DENSITIES
# =============================================================================

CONVERSION_NE_RHO_MPCH = 0.0289 * 1e18 / H**2

RHO0_1 = NE0_1 * CONVERSION_NE_RHO_MPCH
RHO0_2 = NE0_2 * CONVERSION_NE_RHO_MPCH


# =============================================================================
# SCHNEIDER ET AL. (2015) PARAMETERS
# =============================================================================

R_200 = 0.77 # Mpc /h

ALPHA_CONTRACTION = 0.68

#Eq 2.20
M_C = 1.2e14  # M_sun / h
BETA = 0.6
EPSILON = 0.023

M1 = 1.526e11  # M_sun / h

ALPHA_SCHNEIDER = -1.779
DELTA = 4.394
GAMMA = 0.547


# =============================================================================
# HALO STRUCTURE
# =============================================================================

R_S = R_200 / config.CONCENTRATION

# Eq. 2.8
R_TRUNCATED = 8.0 * R_200
TAU = R_TRUNCATED / R_S

# =============================================================================
# BARYONIC PROFILE PARAMETERS
# =============================================================================

R_H = 0.015 * R_200  # Half-light radius
R_EJECTED = 3.5 * R_200  # Ejection radius

F_CENTRAL_GALAXY = 0.005

F_RDM = 1 - F_B

# =============================================================================
# MASS FRACTIONS (SCHNEIDER MODEL 2015)
# =============================================================================

if config.SCHNEIDER2015_CASE == 'A':
    F_BOUNDED_GAS = 0.151

elif config.SCHNEIDER2015_CASE == 'B':
    F_BOUNDED_GAS = 0.078

elif config.SCHNEIDER2015_CASE == 'C':
    F_BOUNDED_GAS = 0.005

F_EJECTED_GAS = (1.0 - F_RDM - F_CENTRAL_GALAXY - F_BOUNDED_GAS)

# =============================================================================
# LIMOUSIN PARAMETERS (Abell 1703 Observational Truth)
# =============================================================================
R_CUT = 30.0 / 1000.0 
R_CORE = 1e-5 / 1000.0 
SIGMA_0 = 299.4                # Fiducial velocity dispersion in km/s


# =============================================================================
# INTERPOLATED DATA FILES
# =============================================================================

# Input data from files

# Interpolated
INTERPOLATED_DM = '../Data/Data_random/NFW.csv'
INTERPOLATED_BARYONS = '../Data/Data_random/Stars.csv'

# =============================================================================
# SCHNEIDER ET AL. (2019) PARAMETERS
# =============================================================================

# Stellar parameters (Eq. 2.11)
A_STAR_2019 = 0.09
M1_STAR_2019 = 2.5e11  # M_sun / h
ETA_STAR_2019 = 0.32
ETA_CGA_2019 = 0.6

# Gas parameters (Eq. 2.14, 2.15, 2.16)
THETA_CO_2019 = 0.1
THETA_EJ_2019 = 4.0    # Benchmark Model A-avrg
M_C_2019 = 2.3e13      # M_sun / h (Benchmark Model A-avrg)
MU_2019 = 0.31         # Benchmark Model A-avrg

P_2019 = 0.3 
Q_2019 = 0.707

# Eq 2.19 
A_RELAX_2019 = 0.3
N = 2.0

# =============================================================================
# SCHNEIDER ET AL. (2025) BARYONIC PARAMETERS
# =============================================================================

# Stellar & Fraction Parameters (Eq. 2.21, 2.23)
ZETA_2025 = 1.376
M_STAR_2025 = 2.5e11  # M_sun / h
N_STAR_2025 = 0.025 
ETA_2025 = 0.05
D_ETA_2025 = 0.23
C_IGA_2025 = 0.1

# Hot Gas Profile Parameters (Eq. 2.12, 2.13)
THETA_C_2025 = 0.3
M_C_2025 = 1e13       # M_sun / h
MU_2025 = 0.7
DELTA_2025 = 6.0

ESPSILON_2025 = 8.0
ESPSILON_0_2025 = 4.0
Q0 = 0.075
Q1 = 0.25
Q2 = 0.5 
N_STEP = 3/2 

R_CORE_2025 = 1e-5 # Mpc/ h

LIMIT_RHO_2025 = 500 # number of intervals in quad ? 


# =============================================================================
# GAP AND INTERSECTION ANALYSIS PARAMETERS
# =============================================================================
# Intersection between 2 curves (to estimate where mb / mdm = 1)
GAP_LINE1_X = 'r'
GAP_LINE1_Y = 'mass_dm_initial'

GAP_LINE2_X = 'r' 
GAP_LINE2_Y = 'mass_baryons'

