import numpy as np
import constantes 
import config
import profiles
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.special import gamma, gammainc


def initialize_mass_profiles(r_i, rho_dm_i, f_baryon):
    integrand = 4.0 * np.pi * r_i**2 * rho_dm_i
    M_dm_enclosed = cumulative_trapezoid(integrand, r_i, initial=0.0)
    
    if len(r_i) > 1 and r_i[1] > 0:
        M_dm_enclosed[0] = M_dm_enclosed[1] * (r_i[0] / r_i[1])**3
    
    M_tot_enc = M_dm_enclosed / (1.0 - f_baryon)

    r_f = np.zeros_like(r_i)
    r_f[0] = r_i[0]

    return M_dm_enclosed, M_tot_enc, r_f

# Contraction calculation MAIN ALGO

def contracted_radius(ri, equation, tol=constantes.TOLERANCE, max_extension_factor= constantes.MAX_EXTENSION_FACTOR):
    a = constantes.A_CONTRACT_RADIUS * ri
    b = ri
    
    fa = equation(a)
    fb = equation(b)
    
    while fa * fb > 0 and b < max_extension_factor * ri: # Dilatation
        
        b *= constantes.MAX_EXTENSION_FACTOR
        fb = equation(b)

    diff = (b - a) / ((a + b) / 2.0)
    while diff > tol:
        mid = (a + b) / 2.0
        f_mid = equation(mid)
        
        if f_mid == 0:
            a = b = mid
            break
        elif fa * f_mid < 0:
            b = mid
            fb = f_mid
        else:
            a = mid
            fa = f_mid
        
        diff = (b - a) / ((a + b) / 2.0)
        
    return (a + b) / 2.0

# Blumenthal and al. (1986)
def contract_blumenthal(r_i, rho_dm_i, mass_b_final_func, f_baryon, tol=constantes.TOLERANCE):
    M_dm_enclosed, M_tot_i, r_f = initialize_mass_profiles(r_i, rho_dm_i, f_baryon)

    for index in range(1, len(r_i)):
        ri = r_i[index]
        mdm_i = M_dm_enclosed[index]
        L_init = ri * M_tot_i[index]

        def equation(rf):
            return rf * (mdm_i + mass_b_final_func(rf)) - L_init

        r_f[index] = contracted_radius(ri, equation, tol=tol)

    return r_f, M_dm_enclosed

#Ando and al. (2015) 
def contract_ando(r_i, rho_dm_i, mass_stars_final_func, mass_gas_final_func, f_baryon, r_vir, A0, W, tol=constantes.TOLERANCE):
    
    M_dm_enclosed, M_tot_enc, r_f = initialize_mass_profiles(r_i, rho_dm_i, f_baryon)

    r0 = 0.03 * r_vir

    # Characteristic orbital radius (Ando 2004)
    def r_bar(r):
        return A0 * r0 * (r / r0) ** W

    M_dm_interp  = interp1d(r_i, M_dm_enclosed, bounds_error=False, fill_value=(M_dm_enclosed[0], M_dm_enclosed[-1]))
    M_tot_interp = interp1d(r_i, M_tot_enc, bounds_error=False, fill_value=(M_tot_enc[0], M_tot_enc[-1]))

    for index in range(1, len(r_i)):
        ri = r_i[index]
        rbar_i = r_bar(ri)

        mdm_bar_i = float(M_dm_interp(rbar_i))
        mtot_bar_i = float(M_tot_interp(rbar_i))
        
        L_init = ri * mtot_bar_i

        def equation(rf):
            rbar_f = r_bar(rf)
            mb_f = mass_stars_final_func(rbar_f) + mass_gas_final_func(rbar_f)
            return rf * (mdm_bar_i + mb_f) - L_init

        r_f[index] = contracted_radius(ri, equation, tol=tol)

    return r_f, M_dm_enclosed

# Schneider et al 2015

def contract_Schneider2015():
    rho_i_dm = profiles.get_schneider_dm_density(config.R) 
    M_i_dm = profiles.enclosed_mass(config.R, rho_i_dm)

    rho_central_galaxy = profiles.rho_central_galaxy(config.R)
    M_central_galaxy = profiles.enclosed_mass(config.R, rho_central_galaxy)

    rho_ejected_gas = profiles.rho_ejected_gas(config.R)
    M_ejected_gas = profiles.enclosed_mass(config.R, rho_ejected_gas)

    rho_bounded_gas = profiles.rho_bounded_gas(config.R)
    M_bounded_gas = profiles.enclosed_mass(config.R, rho_bounded_gas)

    initial_components = {
        "rho_i_dm": rho_i_dm,
        "M_i_dm": M_i_dm,
        "rho_central_galaxy": rho_central_galaxy,
        "M_central_galaxy": M_central_galaxy,
        "rho_ejected_gas": rho_ejected_gas,
        "M_ejected_gas": M_ejected_gas,
        "rho_bounded_gas": rho_bounded_gas,
        "M_bounded_gas": M_bounded_gas
    }
    r_f = np.zeros_like(config.R)
    r_f[0] = config.R[0]

    m_cg_f_array = np.zeros_like(config.R)
    m_eg_f_array = np.zeros_like(config.R)
    m_bg_f_array = np.zeros_like(config.R)
    m_baryon_f_array = np.zeros_like(config.R)
    m_tot_f_array = np.zeros_like(config.R)
    L_f_array = np.zeros_like(config.R)
    ratio_mf_array = np.zeros_like(config.R)
    ratio_r_array = np.zeros_like(config.R)

    for index in range(1, len(config.R)): 
        ri = config.R[index]
        Mi = M_i_dm[index]
        
        def equation(rf):
            mf_central_galaxy = np.interp(rf, config.R, M_central_galaxy)
            mf_bounded_gas = np.interp(rf, config.R, M_bounded_gas)
            mf_ejected_gas = np.interp(rf, config.R, M_ejected_gas)
                
            Mf = constantes.F_RDM * Mi + constantes.F_CENTRAL_GALAXY * mf_central_galaxy + constantes.F_BOUNDED_GAS * mf_bounded_gas + constantes.F_EJECTED_GAS * mf_ejected_gas
            
            return (rf / ri) - 1.0 - constantes.ALPHA_CONTRACTION * ((Mi / Mf) - 1) 
        
        r_f[index] = contracted_radius(ri, equation)
        rf = r_f[index]

        m_cg_f_array[index] = np.interp(rf, config.R, M_central_galaxy)
        m_eg_f_array[index] = np.interp(rf, config.R, M_ejected_gas)
        m_bg_f_array[index] = np.interp(rf, config.R, M_bounded_gas)
        
        m_baryon_f_array[index] = (constantes.F_CENTRAL_GALAXY * m_cg_f_array[index] + constantes.F_BOUNDED_GAS * m_bg_f_array[index] + constantes.F_EJECTED_GAS * m_eg_f_array[index])
        
        m_tot_f_array[index] = constantes.F_RDM * Mi + m_baryon_f_array[index]
        
        L_f_array[index] = rf * m_tot_f_array[index]
        ratio_mf_array[index] = m_baryon_f_array[index] / Mi
        ratio_r_array[index] = rf / ri

    M_rdm_grid = np.interp(config.R, r_f, M_i_dm)

    dM_dr = np.gradient(M_rdm_grid, config.R)
    rho_rdm = dM_dr / (4.0 * np.pi * config.R**2)

    rho_baryons = constantes.F_CENTRAL_GALAXY * rho_central_galaxy + constantes.F_BOUNDED_GAS * rho_bounded_gas + constantes.F_EJECTED_GAS * rho_ejected_gas
    rho_bcm = constantes.F_RDM * rho_rdm + rho_baryons

    return rho_bcm, rho_rdm, rho_baryons, initial_components, m_baryon_f_array, r_f


# Schneider et al (2015)
def contract_Schneider2015_general(r_i, rho_dm_i, mass_stars_final_func, mass_gas_final_func, f_baryon,
                       alpha_contraction=constantes.ALPHA_CONTRACTION,tol=constantes.TOLERANCE):

    M_dm_enclosed, _, r_f = initialize_mass_profiles(r_i, rho_dm_i, f_baryon)
    f_rdm = 1.0 - f_baryon

    for index in range(1, len(r_i)):
        ri    = r_i[index]
        Mi_dm = M_dm_enclosed[index]
        Mi_eq = Mi_dm / f_rdm  

        def equation(rf):
            mb_f = mass_stars_final_func(rf) + mass_gas_final_func(rf)
            Mf   = Mi_dm + mb_f
            return (rf / ri) - 1.0 - alpha_contraction * (Mi_eq / Mf - 1)

        r_f[index] = contracted_radius(ri, equation, tol=tol)

    return r_f, M_dm_enclosed


# Schneider and al. (2019) - Eq. 2.17
def contract_Schneider2019(r_i, rho_dm_i, mass_stars_final_func, mass_gas_final_func, f_baryon,
                                   a=constantes.A_RELAX_2019, n=constantes.N, tol=constantes.TOLERANCE):
        
    M_dm_enclosed, M_tot_enc, r_f = initialize_mass_profiles(r_i, rho_dm_i, f_baryon)

    for index in range(1, len(r_i)):
        ri = r_i[index]
        mdm_i = M_dm_enclosed[index]
        Mi = M_tot_enc[index] 

        def equation(rf):
            mb_f = mass_stars_final_func(rf) + mass_gas_final_func(rf)
            Mf = mdm_i + mb_f 
            
            if Mf == 0:
                return rf - ri
                
            return (rf / ri) - 1.0 - a * ((Mi / Mf)**n - 1.0)

        r_f[index] = contracted_radius(ri, equation, tol=tol)

    return r_f, M_dm_enclosed


# Schneider et al 2025
# Eq B 10 effective calculation no iterative process
# To be done:  compare with B 5, B 6, B 7
def contract_Schneider2025(r_i, M_nfw_i, M_cga_i, M_iga_i, M_hga_i, f_cga, f_iga, f_hga, r_200, 
                               epsilon=constantes.ESPSILON_2025, epsilon_0=constantes.ESPSILON_0_2025, 
                               Q0=constantes.Q0, Q1=constantes.Q1, Q2=constantes.Q2, n_step=constantes.N_STEP):

    r_step = (epsilon / epsilon_0) * r_200

    M_nfw_safe = np.where(M_nfw_i == 0, constantes.NUMERICAL_SAFETY, M_nfw_i)
    
    ratio_cga = np.minimum(M_cga_i / M_nfw_safe, 100.0)
    ratio_iga = np.minimum(M_iga_i / M_nfw_safe, 100.0)
    ratio_hga = np.minimum(M_hga_i / M_nfw_safe, 100.0)
    
    term1 = Q0 / (1.0 + (r_i / r_step)**n_step)
    term2 = Q1 * (ratio_cga - f_cga)
    term3 = Q1 * (ratio_iga - f_iga)
    term4 = Q2 * (ratio_hga - f_hga)
    
    rhs = term1 + term2 + term3 + term4
    
    r_f = r_i / (1.0 + rhs)
    
    r_f[0] = r_i[0]
    for i in range(1, len(r_f)):
        if r_f[i] <= r_f[i-1]:
            r_f[i] = r_f[i-1] + 1e-10
            
    return r_f