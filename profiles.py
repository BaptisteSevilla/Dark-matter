import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.special import gamma, gammainc
import constantes
import config

class Halo:
    def __init__(self, halo_mass, concentration, rho_c, z, omega_m, omega_l, alpha):
        self.halo_mass = halo_mass
        self.c = concentration
        
        self.rho_c = rho_c 
        self.z = z
        
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.alpha = alpha
        
        self.delta_vir = self._calculate_delta_vir()
        self.r_vir = self._calculate_r_vir()
        self.r_scale = self.r_vir / self.c
        
        self.rho_0_nfw = self._calculate_rho_0_nfw()
        self.rho_2_einasto = self._calculate_rho_2_einasto()
        self.rho_0_burkert = self._calculate_rho_0_burkert()
        self.rho_0_custom = self._calculate_rho_0_custom()

    def _calculate_delta_vir(self):
        d = (self.omega_m * (1.0 + self.z)**3) / (self.omega_m * (1.0 + self.z)**3 + self.omega_l) - 1.0
        # Bryan and Norman (1998)
        return 18.0 * np.pi**2 + 82.0 * d - 39.0 * d**2

    def _calculate_r_vir(self):
        return constantes.DELTA_VIR

    def _calculate_rho_0_nfw(self):
        return nfw_rho_s(self.halo_mass, self.r_scale, self.c)

    def _calculate_rho_2_einasto(self):  # ref pour les equations 
        a = 3.0 / self.alpha
        x = (2.0 / self.alpha) * (self.c**self.alpha)
        
        inc_gamma = gammainc(a, x) * gamma(a)
        
        term1 = (4.0 * np.pi * self.r_scale**3) / self.alpha
        term2 = np.exp(2.0 / self.alpha)
        term3 = (2.0 / self.alpha)**(-3.0 / self.alpha)
        
        denominator = term1 * term2 * term3 * inc_gamma
        
        return self.halo_mass / denominator

    def _calculate_rho_0_burkert(self):
        return burkert_rho_0(self.halo_mass, self.r_scale, self.c)

    def density_nfw(self, r):
        return nfw_density(r, self.rho_0_nfw, self.r_scale)

    def density_einasto(self, r):
        r = np.asarray(r)  
        exponent = -(2.0 / self.alpha) * ((r / self.r_scale)**self.alpha - 1.0)
        return self.rho_2_einasto * np.exp(exponent)

    def density_burkert(self, r):
        return burkert_density(r, self.rho_0_burkert, self.r_scale)

    def mass_nfw(self, r):
        return nfw_enclosed_mass(r, self.halo_mass, self.r_scale, self.c)
    
    def mass_einasto(self, r):
        r = np.asarray(r)
        a = 3.0 / self.alpha
        x_r = (2.0 / self.alpha) * ((r / self.r_scale)**self.alpha)
        x_tot = (2.0 / self.alpha) * (self.c**self.alpha)
        return self.halo_mass * (gammainc(a, x_r) / gammainc(a, x_tot))
    
    def mass_burkert(self, r):
        return burkert_enclosed_mass(r, self.halo_mass, self.r_scale, self.c)
    
    def _calculate_rho_0_custom(self):
        alpha = config.ALPHA_CUSTOM
        beta = config.BETA_CUSTOM
        gamma = config.GAMMA_CUSTOM

        def integrand(x):
            x_safe = np.where(x == 0, constantes.NUMERICAL_SAFETY, x)
            return (x_safe**2) / ((x_safe**gamma) * (1.0 + x_safe**alpha)**((beta - gamma) / alpha))
        
        integral_val, _ = quad(integrand, constantes.NUMERICAL_SAFETY, self.c)
        
        rho_0 = self.halo_mass / (4.0 * np.pi * (self.r_scale**3) * integral_val)
        return rho_0
    
    def density_custom(self, r):
        r_s = self.r_vir / self.c
        x = np.asarray(r) / r_s
        x = np.where(x == 0, constantes.NUMERICAL_SAFETY, x)
        
        return self.rho_0_custom / ((x**config.GAMMA_CUSTOM) * (1.0 + x**config.ALPHA_CUSTOM)**((config.BETA_CUSTOM - config.GAMMA_CUSTOM) / config.ALPHA_CUSTOM))

    def mass_custom(self, r):
        r_arr = np.asarray(r)
        
        if r_arr.ndim == 0:
            integrand = lambda x: 4.0 * np.pi * x**2 * self.density_custom(x)
            mass, _ = quad(integrand, constantes.NUMERICAL_SAFETY, r_arr)
            return mass
            
        else:
            density_arr = self.density_custom(r_arr)
            integrand = 4.0 * np.pi * r_arr**2 * density_arr
            mass_arr = np.zeros_like(r_arr)
            mass_arr[1:] = cumulative_trapezoid(integrand, r_arr)
            return mass_arr

    def density(self, r):
        if constantes.DM_PROFILE == 'NFW':
            return self.density_nfw(r)
        elif constantes.DM_PROFILE == 'Einasto':
            return self.density_einasto(r)
        elif constantes.DM_PROFILE == 'Burkert':
            return self.density_burkert(r)
        elif constantes.DM_PROFILE == 'Custom':
            return self.density_custom(r)

    def mass(self, r):
        if constantes.DM_PROFILE == 'NFW':
            return self.mass_nfw(r)
        elif constantes.DM_PROFILE == 'Einasto':
            return self.mass_einasto(r)
        elif constantes.DM_PROFILE == 'Burkert':
            return self.mass_burkert(r)
        elif constantes.DM_PROFILE == 'Custom':
            return self.mass_custom(r)

class PIEMD: # REf de Marceau

    def __init__(self, sigma_0, r_core, r_cut, G=constantes.G):
        self.sigma_0 = sigma_0
        self.r_core  = r_core
        self.r_cut   = r_cut
        self.G       = G
        self.rho_0   = self._calculate_rho_0()

    # Eq. 8
    def _calculate_rho_0(self):
        term1 = self.sigma_0**2 / (2.0 * np.pi * self.G)
        term2 = (self.r_cut + self.r_core) / (self.r_core**2 * self.r_cut)
        return term1 * term2

    # Eq. 6
    def density(self, r):
        r = np.asarray(r)
        return self.rho_0 / ((1.0 + (r / self.r_core)**2) * (1.0 + (r / self.r_cut)**2))

    def enclosed_mass_3d(self, r):
        r  = np.asarray(r)
        r_core, r_cut = self.r_core, self.r_cut
        prefactor = 4.0 * np.pi * self.rho_0 * r_core**2 * r_cut**2 / (r_cut**2 - r_core**2)
        return prefactor * (r_cut * np.arctan(r / r_cut) - r_core * np.arctan(r / r_core))

    # Eq. 7
    def surface_density(self, R):
        R = np.asarray(R)
        prefactor = (self.sigma_0**2 * self.r_cut) / (2.0 * self.G * (self.r_cut - self.r_core))
        return prefactor * (1.0 / np.sqrt(self.r_core**2 + R**2)- 1.0 / np.sqrt(self.r_cut**2  + R**2))

    # Eq. 9
    def aperture_mass(self, R):
        R = np.asarray(R)
        prefactor = (np.pi * self.r_cut * self.sigma_0**2) / self.G
        num = np.sqrt(self.r_cut**2 + R**2) - np.sqrt(self.r_core**2 + R**2)
        den = self.r_cut - self.r_core
        return prefactor * (1.0 - num / den)

    # Eq. 10
    def total_projected_mass(self):
        return (np.pi * self.sigma_0**2 / self.G) * (self.r_cut**2 / (self.r_cut + self.r_core))
    

# =============================================================================
# Dark matter profile
# =============================================================================

def nfw_rho_s(halo_mass, r_s, concentration): # Mettre les refs 
    norm = np.log(1.0 + concentration) - concentration / (1.0 + concentration)
    return halo_mass / (4.0 * np.pi * r_s**3 * norm)

def nfw_density(r, rho_s, r_s):
    r = np.asarray(r)
    return rho_s / ((r / r_s) * (1.0 + r / r_s)**2)

def nfw_enclosed_mass(r, total_mass, r_scale, concentration):
    r = np.asarray(r)
    fc = np.log(1.0 + concentration) - concentration / (1.0 + concentration)
    fx = np.log(1.0 + r / r_scale) - (r / r_scale) / (1.0 + r / r_scale)
    return total_mass * fx / fc

def burkert_mass(x):
    return 0.25 * (np.log(1.0 + x**2) + 2.0 * np.log(1.0 + x) - 2.0 * np.arctan(x))

def burkert_rho_0(halo_mass, r_s, concentration):
    norm = burkert_mass(concentration)
    return halo_mass / (4.0 * np.pi * r_s**3 * norm)

def burkert_density(r, rho_0, r_s):
    r = np.asarray(r)
    x = r / r_s
    return rho_0 / ((1.0 + x) * (1.0 + x**2))

def burkert_enclosed_mass(r, total_mass, r_s, concentration):
    r = np.asarray(r)
    x = r / r_s
    fc = burkert_mass(concentration)
    fx = burkert_mass(x)
    return total_mass * fx / fc

def dimensionless_profile(x, alpha, beta, gamma):
    x = np.where(x == 0, constantes.NUMERICAL_SAFETY, x)
    return 1.0 / ((x**gamma) * (1 + x**alpha)**((beta - gamma) / alpha))

def calc_density_fixed_mass(r, M_vir, c, R_vir, alpha, beta, gamma):
    r_s = R_vir / c

    def integrand(x):
        return (x**2) * dimensionless_profile(x, alpha, beta, gamma)
    
    integral_val, _ = quad(integrand, constantes.NUMERICAL_SAFETY, c)
    rho_0 = M_vir / (4 * np.pi * (r_s**3) * integral_val)
    x = r / r_s
    return rho_0 * dimensionless_profile(x, alpha, beta, gamma)

# Schnneider et al. (2015)
def mass_nfw_truncated(x, tau):
    # Eq A.2
    t1 = x * (x - 2*tau**6 + x*(1-3*x)*tau**4 + x**2 + 2*(1+x-x**2)*tau**2) / ((1+x)*(tau**2+x**2))
    t2 = tau * (6*tau**2 - 2) * np.arctan(x/tau)
    t3 = tau**2 * (tau**2 - 3) * np.log(tau**2 * (1+x)**2 / (tau**2 + x**2))
    return (tau**2 / (2*(1+tau**2)**3)) * (t1 + t2 + t3)

def mass_nfw_truncated_infinity(tau):
    # Eq A.4
    num = 2*tau**4 * (tau**2 - 3)*np.log(tau) - tau**2*(3*tau**2 - 1)*(tau**2 - np.pi*tau + 1)
    den = 2*(1+tau**2)**3
    return num / den

RHO_0 = config.HALO_MASS / (4.0 * np.pi * constantes.R_S**3 * mass_nfw_truncated(config.CONCENTRATION, constantes.TAU))
M_TOT = 4.0 * np.pi * RHO_0 * constantes.R_S**3 * mass_nfw_truncated_infinity(constantes.TAU) 

def rho_nfw_truncated(r):
    x = r / constantes.R_S
    return RHO_0 / (x * (1.0 + x)**2) * (1.0 / (1.0 + (x / constantes.TAU)**2)**2)

# =============================================================================
# Baryons profiles 
# =============================================================================

def hernquist_density(r, m_b, r_b):
    return (m_b * r_b) / (2.0 * np.pi * r * (r + r_b)**3)

def hernquist_mass(r, m_b, r_b):
    return m_b * r**2 / (r + r_b)**2

def disk_density(r, m_b, r_b):
    return (m_b / (4.0 * np.pi * r_b**2 * r)) * np.exp(-r / r_b)

def disk_mass(r, m_b, r_b):
    return m_b * (1.0 - np.exp(-r / r_b) * (1.0 + r / r_b))

# Equation 4.1, Ando & Nagai (2012)
def nuker_surface_brightness(r):
    x = r / constantes.R_B_MPCH
    prefactor = (2.0 ** ((constantes.BETA_FORNAX - constantes.GAMMA_FORNAX) / constantes.ALPHA_FORNAX) * constantes.I_B_LIN_MPCH)
    inner = x ** (-constantes.GAMMA_FORNAX)
    outer = (1.0 + x ** constantes.ALPHA_FORNAX) ** ((constantes.GAMMA_FORNAX - constantes.BETA_FORNAX) / constantes.ALPHA_FORNAX)
    return prefactor * inner * outer

def _get_surface_brightness_derivative(r):
    epsilon = max(r * constantes.ESPSILON_BRIGHTNESS_MIN, constantes.ESPSILON_BRIGHTNESS_MAX)
    return (nuker_surface_brightness(r + epsilon) - nuker_surface_brightness(r - epsilon)) / (2.0 * epsilon)

# Equation 4.2, Ando & Nagai (2012)
def abell_inversion(r):
    def integrand(u):
        R = np.sqrt(u * u + r * r)
        return _get_surface_brightness_derivative(R) / R

    epsrel = constantes.ABEL_EPSREL if r < constantes.ABEL_EPSREL_MIN  else constantes.ABEL_EPSREL_MAX
    val, _ = quad(integrand, 0.0, np.inf, limit=constantes.ABEL_LIMIT, epsrel=epsrel, epsabs= constantes.ABEL_EPSABS)
    return -(1.0 / np.pi) * val


def _precompute_nuker_profiles(r_grid):
    j_arr = np.array([abell_inversion(ri) for ri in r_grid])

    # Convert light to mass density [M_sun/kpc^3]
    rho_arr = constantes.M_L_RATIO * j_arr  
    
    mass_arr = cumulative_trapezoid(4.0 * np.pi * r_grid**2 * rho_arr, r_grid, initial=0.0)  
    
    return rho_arr, mass_arr

_nuker_rho_data, _nuker_mass_data = _precompute_nuker_profiles(config.R)

_interp_nuker_density = interp1d(config.R, _nuker_rho_data, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0))
_interp_nuker_mass = interp1d(config.R, _nuker_mass_data, kind='cubic', bounds_error=False, fill_value=(0.0, _nuker_mass_data[-1]))

def nuker_density(r):
    return _interp_nuker_density(r)

def nuker_mass(r):
    return _interp_nuker_mass(r)

def exact_gas_density(r):
    dens_central = constantes.RHO0_1 * (1.0 + (r / constantes.R_C1_MPCH)**2)**(-1.5 * constantes.BETA1_GAS)
    dens_galactic = constantes.RHO0_2 * (1.0 + (r / constantes.R_C2_MPCH)**2)**(-1.5 * constantes.BETA2_GAS)
    return dens_central + dens_galactic

def exact_gas_mass(r_target):
    def integrand(r):
        return 4.0 * np.pi * r**2 * exact_gas_density(r)
    mass, _ = quad(integrand, 0.0, r_target)
    return mass 

# Schneider et al (2015)

def rho_ejected_gas(r):
    # Eq. 2.13 normalized by M_TOT
    return (M_TOT / (2 * np.pi * constantes.R_EJECTED**2)**(3/2)) * np.exp(-r**2 / (2 * constantes.R_EJECTED**2))

def rho_central_galaxy(r):
    # Eq. 2.14 normalized by M_TOT
    return (M_TOT / (4 * np.pi**(3/2) * constantes.R_H * r**2)) * np.exp(-(r / (2*constantes.R_H))**2)

def rho_bounded_gas(r):
    # Eq. 2.10
    r = np.asarray(r)
    rho = np.zeros_like(r)
    
    r_star = constantes.R_200 / np.sqrt(5.0)
    x_star = r_star / constantes.R_S
    
    # Eq. 2.11
    x_c = config.CONCENTRATION / np.sqrt(5.0)
    gamma = ((1.0 + 3.0*x_c) * np.log(1.0 + x_c)) / ((1.0 + x_c) * np.log(1.0 + x_c) - x_c)
    
    # Eq. 2.10
    rho_nfw_star = rho_nfw_truncated(r_star)
    base_star = np.log(1.0 + x_star) / x_star
    y0 = rho_nfw_star / (base_star ** gamma)
    
    core_mask = r < r_star
    x_core = r[core_mask] / constantes.R_S
    
    base = np.where(x_core < 1e-4, 1.0 - x_core/2.0, np.log(1.0 + x_core) / x_core)
    
    rho[core_mask] = y0 * (base ** gamma)
    rho[~core_mask] = rho_nfw_truncated(r[~core_mask])
    
    return rho

# Schneider et al (2019)

def get_fractions_schneider2019(M_200):
    f_star = constantes.A_STAR_2019 * (constantes.M1_STAR_2019 / M_200)**constantes.ETA_STAR_2019
    f_cga = constantes.A_STAR_2019 * (constantes.M1_STAR_2019 / M_200)**constantes.ETA_CGA_2019
    f_sga = f_star - f_cga
    
    f_gas = (constantes.OMEGA_B / constantes.OMEGA_M) - f_star
    f_clm = 1.0 - (constantes.OMEGA_B / constantes.OMEGA_M) + f_sga
    
    return f_star, f_cga, f_sga, f_gas, f_clm

def beta_schneider2019(M_200):
    return 3.0 - (constantes.M_C / M_200)**constantes.MU_2019

def rho_cga_schneider2019(r, M_tot, r_200, f_cga):
    r_arr = np.asarray(r)
    r_safe = np.where(r_arr == 0, constantes.NUMERICAL_SAFETY, r_arr)
    
    R_h = 0.015 * r_200
    prefactor = (f_cga * M_tot) / (4.0 * np.pi**(3.0/2.0) * R_h)
    
    return prefactor * (1.0 / r_safe**2) * np.exp(-(r_safe / (2.0 * R_h))**2)

def rho_gas_schneider2019(r, M_tot, r_200, f_gas, beta):
    r_co = constantes.THETA_CO_2019 * r_200
    r_ej = constantes.THETA_EJ_2019 * r_200
    
    def unnormalized_density(x):
        u = x / r_co
        v = x / r_ej
        return 1.0 / ((1.0 + u)**beta * (1.0 + v**2)**((7.0 - beta) / 2.0))
    
    integral_val, _ = quad(lambda x: x**2 * unnormalized_density(x), 0, np.inf)
    rho_gas_0 = (f_gas * M_tot) / (4.0 * np.pi * integral_val)
    
    return rho_gas_0 * unnormalized_density(np.asarray(r))

def schneider2019_baryon_fractions(M_200):    
    f_star = constantes.A_STAR_2019 * (constantes.M1_STAR_2019 / M_200)**constantes.ETA_STAR_2019
    f_cga = constantes.A_STAR_2019 * (constantes.M1_STAR_2019 / M_200)**constantes.ETA_CGA_2019
    f_gas = constantes.F_B - f_star
    
    return f_cga, f_gas

def rho_cga_2019(r, M_200, R_200):
    f_cga, _ = schneider2019_baryon_fractions(M_200)
    R_h = 0.015 * R_200
    r_arr = np.asarray(r)
    r_safe = np.where(r_arr == 0, constantes.NUMERICAL_SAFETY, r_arr)
    
    prefactor = (f_cga * M_200) / (4.0 * np.pi**(1.5) * R_h)
    return prefactor * (1.0 / r_safe**2) * np.exp(-(r_safe / (2.0 * R_h))**2)

def rho_gas_2019(r, M_200, R_200):
    _, f_gas = schneider2019_baryon_fractions(M_200)
    
    m_c_2019 = constantes.M_C_2019
    mu_2019 = constantes.MU_2019
    theta_co_2019 = constantes.THETA_CO_2019
    theta_ej_2019 = constantes.THETA_EJ_2019
    
    beta = 3.0 - (m_c_2019 / M_200)**mu_2019
    r_co = theta_co_2019 * R_200
    r_ej = theta_ej_2019 * R_200
    
    r_arr = np.asarray(r)
    
    def shape_func(radius):
        u = radius / r_co
        v = radius / r_ej
        return 1.0 / ((1.0 + u)**beta * (1.0 + v**2)**((7.0 - beta) / 2.0))
        
    integrand = lambda x: x**2 * shape_func(x)
    integral_val, _ = quad(integrand, 0, np.inf, limit=500)
    
    rho_gas_0 = (f_gas * M_200) / (4.0 * np.pi * integral_val)
    return rho_gas_0 * shape_func(r_arr)

# Schneider et al 2025

def schneider2025_baryon_fractions(M_200):
    # Eq. 2.21
    term_star = (M_200 / constantes.M_STAR_2025)**(-constantes.ETA_2025) + (M_200 / constantes.M_STAR_2025)**(constantes.ETA_2025)
    f_star = constantes.N_STAR_2025 * term_star**(-constantes.ZETA_2025)
    
    term_cga = (M_200 / constantes.M_STAR_2025)**(-constantes.ETA_2025) + (M_200 / constantes.M_STAR_2025)**(constantes.ETA_2025 + constantes.D_ETA_2025)
    f_cga = constantes.N_STAR_2025 * term_cga**(-constantes.ZETA_2025)
    
    # Eq. 2.22, 2.23, 2.24
    f_sga = f_star - f_cga
    f_iga = constantes.C_IGA_2025 * f_cga
    f_hga = constantes.F_B - f_star - f_iga
    
    return f_star, f_cga, f_sga, f_iga, f_hga

def y_hga_2025(r, M_200, R_200):
    # Eq. 2.12
    r_c = constantes.THETA_C_2025 * R_200
    r_t = constantes.R_TRUNCATED
    beta = 3.0 * (M_200 / constantes.M_C_2025)**constantes.MU_2025 / (1.0 + (M_200 / constantes.M_C_2025)**constantes.MU_2025)
    r_safe = np.where(r == 0, constantes.NUMERICAL_SAFETY, r)
    return (1.0 + (r_safe / r_c))**(-beta) * (1.0 + (r_safe / r_t)**1.5)**(-constantes.DELTA_2025 / 1.5)

def y_iga_2025(r, R_200):
    # Eq. 2.14
    r_safe = np.where(r == 0, constantes.NUMERICAL_SAFETY, r)
    r_core = constantes.R_CORE_2025
    return (1.0 / (r_safe**2 + r_core**2)**1.5) * np.exp(-r_safe / R_200)

def y_cga_2025(r, R_200):
    # Eq. 2.15
    r_safe = np.where(r == 0, constantes.NUMERICAL_SAFETY, r)
    R_cga = 0.03 * R_200
    return (1.0 / r_safe**2) * np.exp(-r_safe / R_cga)

def rho_hga_2025(r, M_200, R_200, f_hga):
    integrand = lambda x: y_hga_2025(x, M_200, R_200) * x**2
    integral_val, _ = quad(integrand, config.R[0], np.inf, limit=constantes.LIMIT_RHO_2025)
    rho_0 = (f_hga * M_TOT) / (4.0 * np.pi * integral_val)
    return rho_0 * y_hga_2025(r, M_200, R_200)

def rho_iga_2025(r, M_200, R_200, f_iga):
    integrand = lambda x: y_iga_2025(x, R_200) * x**2
    integral_val, _ = quad(integrand, config.R[0], np.inf, limit=constantes.LIMIT_RHO_2025)
    rho_0 = (f_iga * M_TOT) / (4.0 * np.pi * integral_val)
    return rho_0 * y_iga_2025(r, R_200)

def rho_cga_2025(r, M_200, R_200, f_cga):
    integrand = lambda x: y_cga_2025(x, R_200) * x**2
    integral_val, _ = quad(integrand, config.R[0], np.inf, limit=constantes.LIMIT_RHO_2025)
    rho_0 = (f_cga * M_TOT) / (4.0 * np.pi * integral_val)
    return rho_0 * y_cga_2025(r, R_200)


# =============================================================================
# Functions for calculating enclosed mass and contracted dark matter density
# =============================================================================

def enclosed_mass(r_arr, rho_arr):
    integrand = 4.0 * np.pi * r_arr**2 * rho_arr
    M = np.zeros_like(r_arr)
    M[1:] = cumulative_trapezoid(integrand, r_arr)
    return M

# Apres la contraction point important 
# mapping des densités avant et apres contraction via drf / dri  
# M dm (rf) = M dm (ri) 
# M dm (rf) = int 0 à rf (rho_dm_c * r_f **2 * drf)
# M dm (ri) = int 0 à ri (rho_dm_i * r_i **2 * dri)

def dm_density_contracted(r_i, rho_dm_i, r_f): # Comparer avec la formule issue des masses directement 
    dr_f_dr_i = np.gradient(r_f, r_i)
    dr_f_dr_i = np.where(np.abs(dr_f_dr_i) < 1e-30, 1e-30, dr_f_dr_i)
    rho_contracted = rho_dm_i * (r_i / r_f)**2 / dr_f_dr_i # Mettre les refs / origine : d'ou ca sort. 

    return rho_contracted


# Mass functions

def mb_hernquist(rf): 
    return hernquist_mass(rf, constantes.M_B, constantes.R_B)

def mb_disk(rf):
    return disk_mass(rf, constantes.M_B, constantes.R_B)

def mb_nuker(rf):
    res = nuker_mass(rf)
    return float(res) if np.ndim(rf) == 0 else np.asarray(res)


# Find the interserction between 2 curves

# Expliquer le raisonnement pour la recherche des différences des signes 
def find_intersections(x, y1, y2, x_min=None, x_max=None):
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= (x >= x_min)
    if x_max is not None:
        mask &= (x <= x_max)
        
    x_search = x[mask]
    y1_search = y1[mask]
    y2_search = y2[mask]
    
    if len(x_search) < 2:
        return []

    log_x = np.log10(x_search)
    log_y1 = np.log10(y1_search)
    log_y2 = np.log10(y2_search)
    
    diff = log_y1 - log_y2
    crossings = np.where(np.diff(np.sign(diff)))[0]
    
    intersections = []
    for i in crossings:
        slope_diff = (diff[i+1] - diff[i]) / (log_x[i+1] - log_x[i])
        log_x_int = log_x[i] - diff[i] / slope_diff
        
        slope_y1 = (log_y1[i+1] - log_y1[i]) / (log_x[i+1] - log_x[i])
        log_y_int = log_y1[i] + slope_y1 * (log_x_int - log_x[i])
        
        intersections.append((10**log_x_int, 10**log_y_int))
        
    return intersections

# 2 halo term Schneider 2019
# Eq 2.8
def xi_lin(r, k_arr, P_lin_arr):
    if r == 0:
        integrand = (k_arr**2) * P_lin_arr / (2 * np.pi**2)
    else:
        integrand = (k_arr**2) * P_lin_arr * (np.sin(k_arr * r) / (k_arr * r)) / (2 * np.pi**2)
    return np.trapz(integrand, x=k_arr)

# Eq 2.9
def calculate_nu(R, D_z, k_arr, P_lin_arr):
    delta_c = 1.686 / D_z
    
    kR = k_arr * R
    kR_safe = np.where(kR == 0, constantes.NUMERICAL_SAFETY, kR)
    W_kR = 3.0 * (np.sin(kR_safe) - kR_safe * np.cos(kR_safe)) / (kR_safe**3)
    
    integrand = (k_arr**2) * P_lin_arr * (W_kR**2) / (2 * np.pi**2)
    variance = np.trapz(integrand, x=k_arr)
    
    nu = delta_c * (variance**-0.5)
    return nu, delta_c

# Eq 2.9
def calculate_b_nu(nu, delta_c):
    term1 = (constantes.Q_2019 * nu**2 - 1) / delta_c
    term2 = (2 * constantes.P_2019) / (delta_c * (1 + (constantes.Q_2019 * nu**2)**constantes.P_2019))
    return 1 + term1 + term2

# Eq 2.7
def calculate_rho_2h(r, M_200, D_z, Omega_m, rho_crit, k_arr, P_lin_arr):
    r_200 = (M_200 / ((4.0 / 3.0) * np.pi * 200.0 * rho_crit))**(1.0 / 3.0)
    
    xi_lin_r = xi_lin(r, k_arr, P_lin_arr)
    nu, delta_c = calculate_nu(r_200, D_z, k_arr, P_lin_arr)
    b_nu = calculate_b_nu(nu, delta_c)
    
    return b_nu * xi_lin_r * Omega_m * rho_crit

def calculate_rho_nfw(r, M_200, c, rho_crit):
    r_200 = (M_200 / ((4.0 / 3.0) * np.pi * 200.0 * rho_crit))**(1.0 / 3.0)
    r_s = r_200 / c
    epsilon = 4.0
    r_t = epsilon * r_200
    
    if r == 0:
        return np.inf
        
    x = r / r_s
    y = r / r_t
    
    fc = np.log(1.0 + c) - (c / (1.0 + c))
    rho_nfw_0 = M_200 / (4.0 * np.pi * r_s**3 * fc)
    
    term_nfw = rho_nfw_0 / (x * (1 + x)**2)
    term_truncation = 1.0 / (1 + y**2)**2
    
    return term_nfw * term_truncation

def calculate_rho_dmo(r, M_200, c, D_z, Omega_m, rho_crit, k_arr, P_lin_arr):
    rho_1h = calculate_rho_nfw(r, M_200, c, rho_crit)
    rho_2h_term = calculate_rho_2h(r, M_200, D_z, Omega_m, rho_crit, k_arr, P_lin_arr)
    
    return rho_1h + rho_2h_term, rho_1h, rho_2h_term


# =============================================================================
# Profiles with parameters in config.py
# =============================================================================

def get_initial_dm_density(r_array=config.R, halo_mass=config.HALO_MASS, concentration=config.CONCENTRATION):

    R_200 = (3.0 * halo_mass / (4.0 * np.pi * 200.0 * constantes.RHO_C))**(1.0 / 3.0)
    R_S = R_200 / concentration

    if config.DM_PROFILE == 'NFW':
        rho_s = nfw_rho_s(halo_mass, R_S, concentration)
        return nfw_density(r_array, rho_s, R_S)

    elif config.DM_PROFILE == 'Einasto':
        _h = Halo(halo_mass, concentration, constantes.RHO_C, constantes.Z,
                  constantes.OMEGA_M, constantes.OMEGA_L, constantes.ALPHA_EINASTO)
        return _h.density_einasto(r_array)

    elif config.DM_PROFILE == 'Burkert':
        rho_0_b = burkert_rho_0(halo_mass, R_S, concentration)
        return burkert_density(r_array, rho_0_b, R_S)

    elif config.DM_PROFILE == 'Custom':
        return calc_density_fixed_mass(
            r=r_array,
            M_vir=halo_mass,
            c=concentration,
            R_vir=R_200,
            alpha=config.ALPHA_CUSTOM,
            beta=config.BETA_CUSTOM,
            gamma=config.GAMMA_CUSTOM,
        )

    elif config.DM_PROFILE == 'PIEMD':
            piemd = PIEMD(constantes.SIGMA_0, constantes.R_CORE, constantes.R_CUT)
            return piemd.density(r_array)


def get_schneider_dm_density(r_array=config.R):
    is_schneider_baryon = config.BARYON_PROFILE in ['Schneider2015', 'Schneider2019', 'Schneider2025']
    force_nfw = getattr(config, 'FORCE_TRUNCATED_NFW_FOR_SCHNEIDER', False) and is_schneider_baryon

    if config.DM_PROFILE == 'NFW' or force_nfw:
        return rho_nfw_truncated(r_array)

    rho_base = get_initial_dm_density(r_array)
    trunc    = 1.0 / (1.0 + (r_array / constantes.R_TRUNCATED)**2)**2
    rho_trunc = rho_base * trunc

    idx_200 = np.searchsorted(r_array, constantes.R_200)
    if idx_200 < 2:
        idx_200 = len(r_array) 
    M_enc = np.trapz(4.0 * np.pi * r_array[:idx_200]**2 * rho_trunc[:idx_200],
                     r_array[:idx_200])
    if M_enc > 0:
        rho_trunc *= config.HALO_MASS / M_enc

    return rho_trunc