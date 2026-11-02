import numpy as np
import constantes 
import config
import profiles
import contraction
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def extended_radius(ri, equation, tol=constantes.TOLERANCE, max_extension_factor=constantes.MAX_EXTENSION_FACTOR):
    # La recherche se fait vers l'exterieur (expansion)
    a = ri
    b = max_extension_factor * ri
    
    fa = equation(a)
    fb = equation(b)
    
    # Dilatation de la fenetre de recherche si la solution est plus lointaine
    while fa * fb > 0 and b < 100.0 * max_extension_factor * ri: 
        b *= 2.0
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

def extend_impulsive(r_i, rho_dm_i, mass_stars_final_func, mass_gas_final_func, f_baryon):
    """
    Modele de perte de masse impulsive (ejection instantanee).
    Formule: r_f = r_i * (M_i / M_f)
    """
    M_dm_enclosed, M_tot_i, r_f = contraction.initialize_mass_profiles(r_i, rho_dm_i, f_baryon)
    
    for index in range(1, len(r_i)):
        ri = r_i[index]
        mdm_i = M_dm_enclosed[index]
        Mi = M_tot_i[index]
        
        # Masse finale a la position initiale r_i
        mb_f = mass_stars_final_func(ri) + mass_gas_final_func(ri)
        Mf = mdm_i + mb_f
        
        if Mf <= 0:
            Mf = constantes.NUMERICAL_SAFETY
            
        r_f[index] = ri * (Mi / Mf)
        
    return r_f, M_dm_enclosed

def extend_adiabatic_slow(r_i, rho_dm_i, mass_stars_final_func, mass_gas_final_func, f_baryon, tol=constantes.TOLERANCE):
    """
    Modele d'expansion adiabatique (perte de masse lente).
    Formule: r_i * M_i = r_f * M_f(r_f)
    """
    M_dm_enclosed, M_tot_i, r_f = contraction.initialize_mass_profiles(r_i, rho_dm_i, f_baryon)

    for index in range(1, len(r_i)):
        ri = r_i[index]
        mdm_i = M_dm_enclosed[index]
        L_init = ri * M_tot_i[index]

        def equation(rf):
            # La masse finale baryonique est evaluee au rayon final rf
            mb_f = mass_stars_final_func(rf) + mass_gas_final_func(rf)
            return rf * (mdm_i + mb_f) - L_init

        r_f[index] = extended_radius(ri, equation, tol=tol)

    return r_f, M_dm_enclosed