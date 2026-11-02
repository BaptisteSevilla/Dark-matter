import numpy as np
import matplotlib.pyplot as plt

def calculate_back_reaction_extension(r_i, r_vir, m_nfw, m_cga, m_iga, m_hga, f_cga, f_iga, f_hga, Q0=0.075, Q1=0.25, Q2=0.5, epsilon=4.0, epsilon0=4.0):
    """
    Evaluates the dark matter radial expansion driven by AGN feedback and baryonic blowout,
    using the back-reaction displacement model.
    """
    r_step = (epsilon / epsilon0) * r_vir
    n_step = 1.5

    # Base contraction prior to feedback
    term_Q0 = Q0 / (1.0 + (r_i / r_step)**n_step)

    # Contraction from central galaxy stars and inner cold gas
    term_Q1_cga = Q1 * f_cga * ((m_cga / m_nfw) - 1.0)
    term_Q1_iga = Q1 * f_iga * ((m_iga / m_nfw) - 1.0)

    # Expansion from outer hot gas ejected by AGN feedback (Blowout effect)
    term_Q2_hga = Q2 * f_hga * ((m_hga / m_nfw) - 1.0)

    # Back-reaction ratio xi = r_i / r_f
    xi = 1.0 + term_Q0 + term_Q1_cga + term_Q1_iga + term_Q2_hga

    r_f = r_i / xi
    return r_f

def plot_halo_kinematics(r_i, r_f, rho_i, m_i):
    """
    Generates plots to compare initial versus extended dark matter states, 
    evaluating halo contraction parameters across radius ratios and enclosed mass.
    """
    dr_i = np.gradient(r_i)
    dr_f = np.gradient(r_f)
    
    # Conservation of mass in spherical shells: rho_f * r_f^2 * dr_f = rho_i * r_i^2 * dr_i
    rho_f = rho_i * (r_i / r_f)**2 * (dr_i / dr_f)
    
    # Mass enclosed within the shell's Lagrangian boundary remains constant
    m_f = m_i 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Logarithmic Density Profiles
    ax1.plot(r_i, rho_i, 'k--', label='Initial DMO')
    ax1.plot(r_f, rho_f, 'r-', label='Extended/Contracted')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Density rho(r)')
    ax1.legend()

    # 2. Enclosed Mass Curves
    ax2.plot(r_i, m_i, 'k--', label='Initial Enclosed Mass')
    ax2.plot(r_f, m_f, 'r-', label='Final Enclosed Mass')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Enclosed Mass M(<r)')
    ax2.legend()

    # 3. Radius Ratio Comparisons
    ax3.plot(r_i, r_f / r_i, 'b-')
    ax3.set_xscale('log')
    ax3.set_xlabel('Initial Radius r_i')
    ax3.set_ylabel('Radius Ratio r_f / r_i')
    ax3.axhline(1.0, color='k', linestyle=':')

    plt.tight_layout()
    plt.show()