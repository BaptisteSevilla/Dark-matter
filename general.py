import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

import constantes
import config
import profiles
import contraction 
import interpolated
import extend 

def contract_halo():

    r = config.R
    halo_mass = config.HALO_MASS
    concentration = config.CONCENTRATION
    f_b = constantes.F_B
    m_b = f_b * halo_mass  
    
    keep_gas = getattr(constantes, 'INCLUDE_GAS', getattr(config, 'GAS', False))
    
    include_xray_gas = keep_gas and (config.BARYON_PROFILE == 'Nuker')

    halo = profiles.Halo(
        halo_mass=halo_mass, concentration=concentration,
        rho_c=constantes.RHO_C, z=constantes.Z, omega_m=constantes.OMEGA_M,
        omega_l=constantes.OMEGA_L, alpha=constantes.ALPHA_EINASTO
    )

    # =================================================================
    #  Initial dark matter
    # =================================================================
    if config.DM_PROFILE == 'Interpolated':
        dm_interp = interpolated.InterpolatedProfile(constantes.INTERPOLATED_DM)
        rho_dm_raw = dm_interp(r)
        
        rho_dm = np.nan_to_num(rho_dm_raw, nan=0.0) * constantes.CONVERSION_FACTOR 
        mass_dm = cumulative_trapezoid(4.0 * np.pi * r**2 * rho_dm, r, initial=0.0)
    else:
        is_schneider_baryon = config.BARYON_PROFILE in ['Schneider2015', 'Schneider2019', 'Schneider2025']
        
        force_nfw = getattr(config, 'FORCE_TRUNCATED_NFW_FOR_SCHNEIDER', False) and is_schneider_baryon and (config.DM_PROFILE == 'NFW')

        if force_nfw:

            rho_dm = (1.0 - f_b) * profiles.rho_nfw_truncated(r)
            mass_dm = profiles.enclosed_mass(r, rho_dm)
        else:
            dispatch_dm = {
                'NFW': (halo.density_nfw, halo.mass_nfw),
                'Einasto': (halo.density_einasto, halo.mass_einasto),
                'Burkert': (halo.density_burkert, halo.mass_burkert),
                'Zhao': (halo.density_custom, halo.mass_custom),
            }
            if config.DM_PROFILE == 'PIEMD':
                _piemd = profiles.PIEMD(
                    sigma_0=constantes.SIGMA_0,
                    r_core=constantes.R_CORE,
                    r_cut=constantes.R_CUT,
                )
                
                piemd_mass_vir = _piemd.enclosed_mass_3d(halo.r_vir)
                norm_factor = halo_mass / piemd_mass_vir
                
                def piemd_density_norm(x):
                    return _piemd.density(x) * norm_factor
                    
                def piemd_mass_norm(x):
                    return _piemd.enclosed_mass_3d(x) * norm_factor

                dispatch_dm['PIEMD'] = (piemd_density_norm, piemd_mass_norm)

            density_fn, mass_fn = dispatch_dm[config.DM_PROFILE]
            rho_dm  = (1.0 - f_b) * density_fn(r)
            mass_dm = (1.0 - f_b) * mass_fn(r)

    # =================================================================
    # Baryons
    # =================================================================
    
    rho_cga = np.zeros_like(r)
    rho_sga = np.zeros_like(r)
    rho_hga = np.zeros_like(r)
    rho_iga = np.zeros_like(r)

    if config.BARYON_PROFILE == 'Interpolated':
        baryons_interp = interpolated.InterpolatedProfile(constantes.INTERPOLATED_BARYONS)
        rho_stars_raw = baryons_interp(r)
        
        rho_cga = np.nan_to_num(rho_stars_raw, nan=0.0) * constantes.CONVERSION_FACTOR 
        mass_stars_arr = cumulative_trapezoid(4.0 * np.pi * r**2 * rho_cga, r, initial=0.0)
        
        mass_stars_func = interp1d(
            r, mass_stars_arr,
            bounds_error=False,
            fill_value=(mass_stars_arr[0], mass_stars_arr[-1])
        )

    elif config.BARYON_PROFILE == 'Hernquist':
        rho_cga = profiles.hernquist_density(r, m_b, constantes.R_B)
        mass_stars_func = profiles.mb_hernquist
        
    elif config.BARYON_PROFILE == 'Disk':
        rho_cga = profiles.disk_density(r, m_b, constantes.R_B)
        mass_stars_func = profiles.mb_disk
        
    elif config.BARYON_PROFILE == 'Nuker':
        rho_cga = profiles.nuker_density(r)
        mass_stars_func = profiles.mb_nuker
        
    elif config.BARYON_PROFILE == 'Schneider2015':
        rho_cga_init = profiles.rho_central_galaxy(r)
        rho_cga = constantes.F_CENTRAL_GALAXY * rho_cga_init
        rho_sga = np.zeros_like(r)

        if keep_gas:
            rho_iga_init = profiles.rho_bounded_gas(r)
            rho_hga_init = profiles.rho_ejected_gas(r)
            rho_iga = constantes.F_BOUNDED_GAS * rho_iga_init
            rho_hga = constantes.F_EJECTED_GAS * rho_hga_init
        else:
            rho_iga = np.zeros_like(r)
            rho_hga = np.zeros_like(r)
        
        rho_baryons_total = rho_cga + rho_iga + rho_hga
        mass_baryons_arr = profiles.enclosed_mass(r, rho_baryons_total)
        
        mass_stars_func = interp1d(
            r, mass_baryons_arr,
            bounds_error=False,
            fill_value=(mass_baryons_arr[0], mass_baryons_arr[-1])
        )
        
    elif config.BARYON_PROFILE == 'Schneider2019':
        R_200 = (3.0 * halo_mass / (4.0 * np.pi * 200.0 * constantes.RHO_C))**(1.0 / 3.0)
        rho_cga = profiles.rho_cga_2019(r, halo_mass, R_200)
        
        mass_stars_arr = profiles.enclosed_mass(r, rho_cga)
        mass_stars_func = interp1d(
            r, mass_stars_arr,
            bounds_error=False,
            fill_value=(mass_stars_arr[0], mass_stars_arr[-1])
        )

    elif config.BARYON_PROFILE == 'Schneider2025':
        R_trunc = halo.r_vir
        f_star, f_cga, f_sga, f_iga, f_hga = profiles.schneider2025_baryon_fractions(halo_mass)
        
        if keep_gas:
            rho_hga = profiles.rho_hga_2025(r, halo_mass, R_trunc, f_hga)
            rho_iga = profiles.rho_iga_2025(r, halo_mass, R_trunc, f_iga)
        else:
            rho_hga = np.zeros_like(r)
            rho_iga = np.zeros_like(r)
            
        gas_mass_arr = profiles.enclosed_mass(r, rho_hga + rho_iga)
        mass_gas_func = interp1d(
            r, gas_mass_arr,
            bounds_error=False,
            fill_value=(gas_mass_arr[0], gas_mass_arr[-1])
        )
        
        rho_cga = profiles.rho_cga_2025(r, halo_mass, R_trunc, f_cga)
        
        cga_mass_arr = profiles.enclosed_mass(r, rho_cga)
        mass_stars_func = interp1d(
            r, cga_mass_arr,
            bounds_error=False,
            fill_value=(cga_mass_arr[0], cga_mass_arr[-1])
        )

    elif config.BARYON_PROFILE == 'PIEMD':
        _piemd = profiles.PIEMD(
            sigma_0=constantes.SIGMA_0,
            r_core=constantes.R_CORE,
            r_cut=constantes.R_CUT
        )
        # 1. Evaluate at physical radius (r / H) to match r_core and r_cut
        # 2. Divide by H**2 to return the density to simulation code units
        rho_cga = _piemd.density(r / constantes.H) / (constantes.H**2)

        # 3. Calculate enclosed mass at physical radius and convert to code units (* H)
        mass_stars_arr = _piemd.enclosed_mass_3d(r / constantes.H) * constantes.H
        
        # 4. Re-add the missing interpolation function required by the contraction algorithm
        mass_stars_func = interp1d(
            r, mass_stars_arr,
            bounds_error=False,
            fill_value=(mass_stars_arr[0], mass_stars_arr[-1])
        )
    # =================================================================
    # 4. Gas
    # =================================================================
    if include_xray_gas:
        gas_mass_arr  = np.array([profiles.exact_gas_mass(ri) for ri in r])
        mass_gas_func = interp1d(
            r, gas_mass_arr,
            bounds_error=False,
            fill_value=(gas_mass_arr[0], gas_mass_arr[-1])
        )
        rho_hga = profiles.exact_gas_density(r)
        
    elif config.BARYON_PROFILE == 'Schneider2019':
        if keep_gas:
            R_200 = (3.0 * halo_mass / (4.0 * np.pi * 200.0 * constantes.RHO_C))**(1.0 / 3.0)
            rho_hga = profiles.rho_gas_2019(r, halo_mass, R_200)
            
            gas_mass_arr = profiles.enclosed_mass(r, rho_hga)
            mass_gas_func = interp1d(
                r, gas_mass_arr,
                bounds_error=False,
                fill_value=(gas_mass_arr[0], gas_mass_arr[-1])
            )
        else:
            rho_hga = np.zeros_like(r)
            mass_gas_func = lambda x: 0.0
        
    elif config.BARYON_PROFILE == 'Schneider2025':
        pass 
        
    elif config.BARYON_PROFILE == 'Schneider2015':
        mass_gas_func = lambda x: 0.0
        
    else:
        mass_gas_func = lambda x: 0.0

    # =================================================================
    # Contraction
    # =================================================================
    if config.CONTRACTION == 'Blumenthal':
        mb_total = lambda x: mass_stars_func(x) + mass_gas_func(x)
        r_f, _ = contraction.contract_blumenthal(r, rho_dm, mb_total, f_b)
        
    elif config.CONTRACTION == 'Ando':
        r_f, _ = contraction.contract_ando(
            r, rho_dm, mass_stars_func, mass_gas_func,
            f_b, halo.r_vir, constantes.A0, constantes.W
        )
        
    elif config.CONTRACTION == 'Schneider2015':  
        r_f, _ = contraction.contract_Schneider2015_general(
            r, rho_dm, mass_stars_func, mass_gas_func,
            f_b, constantes.ALPHA_CONTRACTION
        )
        
    elif config.CONTRACTION == 'Schneider2019':
        r_f, _ = contraction.contract_Schneider2019(
            r_i=r,
            rho_dm_i=rho_dm,
            mass_stars_final_func=mass_stars_func,
            mass_gas_final_func=mass_gas_func,
            f_baryon=f_b
        )
        
    elif config.CONTRACTION == 'Schneider2025':
        f_star, f_cga, f_sga, f_iga, f_hga = profiles.schneider2025_baryon_fractions(halo_mass)
        
        if not keep_gas:
            f_iga = 0.0
            f_hga = 0.0
            
        M_dmo_i = mass_dm / (1.0 - f_b)
        
        if config.BARYON_PROFILE == 'Schneider2025':
            M_cga_arr = profiles.enclosed_mass(r, rho_cga)
            M_iga_arr = profiles.enclosed_mass(r, rho_iga)
            M_hga_arr = profiles.enclosed_mass(r, rho_hga)
        else:
            M_cga_arr = mass_stars_func(r)
            M_hga_arr = mass_gas_func(r)
            M_iga_arr = np.zeros_like(r)
            
        r_f = contraction.contract_Schneider2025(
            r_i=r, 
            M_nfw_i=M_dmo_i, 
            M_cga_i=M_cga_arr, 
            M_iga_i=M_iga_arr, 
            M_hga_i=M_hga_arr, 
            f_cga=f_cga, 
            f_iga=f_iga, 
            f_hga=f_hga, 
            r_200=halo.r_vir
        )
        
    else:
        raise ValueError(f"Unknown CONTRACTION mode: {config.CONTRACTION}")

    # =================================================================
    # Results
    # =================================================================
    rho_dm_contracted = profiles.dm_density_contracted(r, rho_dm, r_f)
    
    if config.BARYON_PROFILE == 'Schneider2025':
        f_star, f_cga, f_sga, f_iga, f_hga = profiles.schneider2025_baryon_fractions(halo_mass)
        # Eq. 2.18
        rho_sga = (f_sga / (1.0 - f_b)) * rho_dm_contracted

    rho_stars = rho_cga + rho_sga
    rho_gas = rho_hga + rho_iga
    rho_baryons  = rho_stars + rho_gas
    rho_total = rho_dm_contracted + rho_baryons
    
    mass_baryons = profiles.enclosed_mass(r, rho_baryons)
    mass_dm_contracted = profiles.enclosed_mass(r, rho_dm_contracted)
    mass_total = profiles.enclosed_mass(r, rho_total)
    
    mass_stars = profiles.enclosed_mass(r, rho_stars)
    mass_gas = profiles.enclosed_mass(r, rho_gas)

    mass_cga = profiles.enclosed_mass(r, rho_cga)
    mass_sga = profiles.enclosed_mass(r, rho_sga)
    mass_hga = profiles.enclosed_mass(r, rho_hga)
    mass_iga = profiles.enclosed_mass(r, rho_iga)

    return {
        'r': r, 'r_f': r_f, 'r_vir': halo.r_vir, 'r_s': halo.r_scale,
        'dm_profile':    config.DM_PROFILE,
        'baryon_profile': config.BARYON_PROFILE,
        'contraction':   config.CONTRACTION,
        'rho_dm_initial': rho_dm / constantes.CONVERSION_FACTOR , 'mass_dm_initial': mass_dm,
        'rho_stars': rho_stars / constantes.CONVERSION_FACTOR , 'rho_gas': rho_gas / constantes.CONVERSION_FACTOR ,
        'rho_baryons': rho_baryons / constantes.CONVERSION_FACTOR , 'mass_baryons': mass_baryons,
        'rho_dm_contracted': rho_dm_contracted / constantes.CONVERSION_FACTOR , 'mass_dm_contracted': mass_dm_contracted,
        'rho_total': rho_total / constantes.CONVERSION_FACTOR , 'mass_total': mass_total,
        
        'mass_stars': mass_stars,
        'mass_gas': mass_gas,
        
        'rho_cga': rho_cga / constantes.CONVERSION_FACTOR,
        'rho_sga': rho_sga / constantes.CONVERSION_FACTOR,
        'rho_hga': rho_hga / constantes.CONVERSION_FACTOR,
        'rho_iga': rho_iga / constantes.CONVERSION_FACTOR,
        'mass_cga': mass_cga,
        'mass_sga': mass_sga,
        'mass_hga': mass_hga,
        'mass_iga': mass_iga
    }


def extend_halo(extension_mode='Impulsive'):
    r = config.R
    halo_mass = config.HALO_MASS
    concentration = config.CONCENTRATION
    f_b = constantes.F_B
    m_b = f_b * halo_mass  
    
    keep_gas = getattr(constantes, 'INCLUDE_GAS', getattr(config, 'GAS', False))
    include_xray_gas = keep_gas and (config.BARYON_PROFILE == 'Nuker')

    halo = profiles.Halo(
        halo_mass=halo_mass, concentration=concentration,
        rho_c=constantes.RHO_C, z=constantes.Z, omega_m=constantes.OMEGA_M,
        omega_l=constantes.OMEGA_L, alpha=constantes.ALPHA_EINASTO
    )

    # =================================================================
    #  Initial dark matter
    # =================================================================
    if config.DM_PROFILE == 'Interpolated':
        dm_interp = interpolated.InterpolatedProfile(constantes.INTERPOLATED_DM)
        rho_dm_raw = dm_interp(r)
        
        rho_dm = np.nan_to_num(rho_dm_raw, nan=0.0) * constantes.CONVERSION_FACTOR 
        mass_dm = cumulative_trapezoid(4.0 * np.pi * r**2 * rho_dm, r, initial=0.0)
    else:
        is_schneider_baryon = config.BARYON_PROFILE in ['Schneider2015', 'Schneider2019', 'Schneider2025']
        force_nfw = getattr(config, 'FORCE_TRUNCATED_NFW_FOR_SCHNEIDER', False) and is_schneider_baryon and (config.DM_PROFILE == 'NFW')

        if force_nfw:
            rho_dm = (1.0 - f_b) * profiles.rho_nfw_truncated(r)
            mass_dm = profiles.enclosed_mass(r, rho_dm)
        else:
            dispatch_dm = {
                'NFW': (halo.density_nfw, halo.mass_nfw),
                'Einasto': (halo.density_einasto, halo.mass_einasto),
                'Burkert': (halo.density_burkert, halo.mass_burkert),
                'Zhao': (halo.density_custom, halo.mass_custom),
            }
            if config.DM_PROFILE == 'PIEMD':
                _piemd = profiles.PIEMD(
                    sigma_0=constantes.SIGMA_0,
                    r_core=constantes.R_CORE,
                    r_cut=constantes.R_CUT,
                )
                piemd_mass_vir = _piemd.enclosed_mass_3d(halo.r_vir)
                norm_factor = halo_mass / piemd_mass_vir
                
                def piemd_density_norm(x):
                    return _piemd.density(x) * norm_factor
                def piemd_mass_norm(x):
                    return _piemd.enclosed_mass_3d(x) * norm_factor

                dispatch_dm['PIEMD'] = (piemd_density_norm, piemd_mass_norm)

            density_fn, mass_fn = dispatch_dm[config.DM_PROFILE]
            rho_dm  = (1.0 - f_b) * density_fn(r)
            mass_dm = (1.0 - f_b) * mass_fn(r)

    # =================================================================
    # Baryons (Final state post-explosion)
    # =================================================================
    
    rho_cga = np.zeros_like(r)
    rho_sga = np.zeros_like(r)
    rho_hga = np.zeros_like(r)
    rho_iga = np.zeros_like(r)

    if config.BARYON_PROFILE == 'Interpolated':
        baryons_interp = interpolated.InterpolatedProfile(constantes.INTERPOLATED_BARYONS)
        rho_stars_raw = baryons_interp(r)
        rho_cga = np.nan_to_num(rho_stars_raw, nan=0.0) * constantes.CONVERSION_FACTOR 
        mass_stars_arr = cumulative_trapezoid(4.0 * np.pi * r**2 * rho_cga, r, initial=0.0)
        mass_stars_func = interp1d(
            r, mass_stars_arr, bounds_error=False, fill_value=(mass_stars_arr[0], mass_stars_arr[-1])
        )

    elif config.BARYON_PROFILE == 'Hernquist':
        rho_cga = profiles.hernquist_density(r, m_b, constantes.R_B)
        mass_stars_func = profiles.mb_hernquist
        
    elif config.BARYON_PROFILE == 'Disk':
        rho_cga = profiles.disk_density(r, m_b, constantes.R_B)
        mass_stars_func = profiles.mb_disk
        
    elif config.BARYON_PROFILE == 'Nuker':
        rho_cga = profiles.nuker_density(r)
        mass_stars_func = profiles.mb_nuker
        
    elif config.BARYON_PROFILE == 'Schneider2015':
        rho_cga_init = profiles.rho_central_galaxy(r)
        rho_cga = constantes.F_CENTRAL_GALAXY * rho_cga_init
        rho_sga = np.zeros_like(r)

        if keep_gas:
            rho_iga_init = profiles.rho_bounded_gas(r)
            rho_hga_init = profiles.rho_ejected_gas(r)
            rho_iga = constantes.F_BOUNDED_GAS * rho_iga_init
            rho_hga = constantes.F_EJECTED_GAS * rho_hga_init
        else:
            rho_iga = np.zeros_like(r)
            rho_hga = np.zeros_like(r)
        
        rho_baryons_total = rho_cga + rho_iga + rho_hga
        mass_baryons_arr = profiles.enclosed_mass(r, rho_baryons_total)
        mass_stars_func = interp1d(
            r, mass_baryons_arr, bounds_error=False, fill_value=(mass_baryons_arr[0], mass_baryons_arr[-1])
        )
        
    elif config.BARYON_PROFILE == 'Schneider2019':
        R_200 = (3.0 * halo_mass / (4.0 * np.pi * 200.0 * constantes.RHO_C))**(1.0 / 3.0)
        rho_cga = profiles.rho_cga_2019(r, halo_mass, R_200)
        mass_stars_arr = profiles.enclosed_mass(r, rho_cga)
        mass_stars_func = interp1d(
            r, mass_stars_arr, bounds_error=False, fill_value=(mass_stars_arr[0], mass_stars_arr[-1])
        )

    elif config.BARYON_PROFILE == 'Schneider2025':
        R_trunc = halo.r_vir
        f_star, f_cga, f_sga, f_iga, f_hga = profiles.schneider2025_baryon_fractions(halo_mass)
        
        if keep_gas:
            rho_hga = profiles.rho_hga_2025(r, halo_mass, R_trunc, f_hga)
            rho_iga = profiles.rho_iga_2025(r, halo_mass, R_trunc, f_iga)
        else:
            rho_hga = np.zeros_like(r)
            rho_iga = np.zeros_like(r)
            
        gas_mass_arr = profiles.enclosed_mass(r, rho_hga + rho_iga)
        mass_gas_func = interp1d(
            r, gas_mass_arr, bounds_error=False, fill_value=(gas_mass_arr[0], gas_mass_arr[-1])
        )
        
        rho_cga = profiles.rho_cga_2025(r, halo_mass, R_trunc, f_cga)
        cga_mass_arr = profiles.enclosed_mass(r, rho_cga)
        mass_stars_func = interp1d(
            r, cga_mass_arr, bounds_error=False, fill_value=(cga_mass_arr[0], cga_mass_arr[-1])
        )

    # =================================================================
    # 4. Gas (Final state post-explosion)
    # =================================================================
    if include_xray_gas:
        gas_mass_arr  = np.array([profiles.exact_gas_mass(ri) for ri in r])
        mass_gas_func = interp1d(
            r, gas_mass_arr, bounds_error=False, fill_value=(gas_mass_arr[0], gas_mass_arr[-1])
        )
        rho_hga = profiles.exact_gas_density(r)
        
    elif config.BARYON_PROFILE == 'Schneider2019':
        if keep_gas:
            R_200 = (3.0 * halo_mass / (4.0 * np.pi * 200.0 * constantes.RHO_C))**(1.0 / 3.0)
            rho_hga = profiles.rho_gas_2019(r, halo_mass, R_200)
            gas_mass_arr = profiles.enclosed_mass(r, rho_hga)
            mass_gas_func = interp1d(
                r, gas_mass_arr, bounds_error=False, fill_value=(gas_mass_arr[0], gas_mass_arr[-1])
            )
        else:
            rho_hga = np.zeros_like(r)
            mass_gas_func = lambda x: 0.0
            
    elif config.BARYON_PROFILE == 'Schneider2025':
        pass 
        
    elif config.BARYON_PROFILE == 'Schneider2015':
        mass_gas_func = lambda x: 0.0
        
    else:
        mass_gas_func = lambda x: 0.0

    # =================================================================
    # Extension (Blowout effect)
    # =================================================================
    if extension_mode == 'Impulsive':
        r_f, _ = extend.extend_impulsive(
            r, rho_dm, mass_stars_func, mass_gas_func, f_b
        )
    elif extension_mode == 'Adiabatic':
        r_f, _ = extend.extend_adiabatic_slow(
            r, rho_dm, mass_stars_func, mass_gas_func, f_b, constantes.TOLERANCE
        )
    else:
        raise ValueError(f"Unknown EXTENSION mode: {extension_mode}")

    # =================================================================
    # Results mapping
    # =================================================================
    rho_dm_extended = profiles.dm_density_contracted(r, rho_dm, r_f)
    
    if config.BARYON_PROFILE == 'Schneider2025':
        f_star, f_cga, f_sga, f_iga, f_hga = profiles.schneider2025_baryon_fractions(halo_mass)
        rho_sga = (f_sga / (1.0 - f_b)) * rho_dm_extended

    rho_stars = rho_cga + rho_sga
    rho_gas = rho_hga + rho_iga
    rho_baryons = rho_stars + rho_gas
    rho_total = rho_dm_extended + rho_baryons
    
    mass_baryons = profiles.enclosed_mass(r, rho_baryons)
    mass_dm_extended = profiles.enclosed_mass(r, rho_dm_extended)
    mass_total = profiles.enclosed_mass(r, rho_total)
    
    mass_stars = profiles.enclosed_mass(r, rho_stars)
    mass_gas = profiles.enclosed_mass(r, rho_gas)

    mass_cga = profiles.enclosed_mass(r, rho_cga)
    mass_sga = profiles.enclosed_mass(r, rho_sga)
    mass_hga = profiles.enclosed_mass(r, rho_hga)
    mass_iga = profiles.enclosed_mass(r, rho_iga)

    return {
        'r': r, 'r_f': r_f, 'r_vir': halo.r_vir, 'r_s': halo.r_scale,
        'dm_profile': config.DM_PROFILE,
        'baryon_profile': config.BARYON_PROFILE,
        'extension_mode': extension_mode,
        'rho_dm_initial': rho_dm / constantes.CONVERSION_FACTOR, 'mass_dm_initial': mass_dm,
        'rho_stars': rho_stars / constantes.CONVERSION_FACTOR, 'rho_gas': rho_gas / constantes.CONVERSION_FACTOR,
        'rho_baryons': rho_baryons / constantes.CONVERSION_FACTOR, 'mass_baryons': mass_baryons,
        'rho_dm_extended': rho_dm_extended / constantes.CONVERSION_FACTOR, 'mass_dm_extended': mass_dm_extended,
        'rho_total': rho_total / constantes.CONVERSION_FACTOR, 'mass_total': mass_total,
        
        'mass_stars': mass_stars,
        'mass_gas': mass_gas,
        
        'rho_cga': rho_cga / constantes.CONVERSION_FACTOR,
        'rho_sga': rho_sga / constantes.CONVERSION_FACTOR,
        'rho_hga': rho_hga / constantes.CONVERSION_FACTOR,
        'rho_iga': rho_iga / constantes.CONVERSION_FACTOR,
        'mass_cga': mass_cga,
        'mass_sga': mass_sga,
        'mass_hga': mass_hga,
        'mass_iga': mass_iga
    }