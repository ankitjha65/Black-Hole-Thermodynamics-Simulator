#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants (SI units)
G = 6.67430e-11  # Gravitational constant
c = 2.99792458e8  # Speed of light
hbar = 1.0545718e-34  # Reduced Planck constant
k_B = 1.380649e-23  # Boltzmann constant
sigma = 5.670374419e-8  # Stefan-Boltzmann constant
M_sun = 1.98847e30  # Solar mass (kg)

# Hawking temperature (K)
def hawking_temperature(M):
    """Calculates Hawking temperature of a black hole."""
    return (hbar * c**3) / (8 * np.pi * G * M * k_B)

# Luminosity (W)
def hawking_luminosity(M):
    """Power emitted via Hawking radiation (Stefan-Boltzmann law)."""
    A = 16 * np.pi * (G * M / c**2)**2  # Black hole surface area
    T = hawking_temperature(M)
    return sigma * A * T**4

# Evaporation time (s)
def evaporation_time(M):
    """Time for a black hole to evaporate completely."""
    # Integrate dM/dt = -P/c² from M to 0
    integrand = lambda M: c**2 / hawking_luminosity(M)
    time, _ = quad(integrand, 0, M)
    return time

# Mass decay over time (kg)
def mass_decay(M_initial, t_max=1e30, steps=1000):
    """Computes mass as a function of time."""
    times = np.logspace(0, np.log10(t_max), steps)
    masses = []
    for t in times:
        if t >= evaporation_time(M_initial):
            masses.append(0)
        else:
            # Approximate mass loss (simplified ODE solution)
            dm_dt = -hawking_luminosity(M_initial) / c**2
            M_initial += dm_dt * (t / steps)
            masses.append(M_initial)
    return times, masses

# Main function
def main():
    # Input: Black hole mass (e.g., 1 solar mass, 1e12 kg, etc.)
    M = float(input("Enter black hole mass (kg): "))
    
    # Calculations
    T = hawking_temperature(M)
    L = hawking_luminosity(M)
    t_evap = evaporation_time(M)
    
    # Convert to readable units
    t_evap_years = t_evap / (60 * 60 * 24 * 365.25)  # years
    
    print("\n--- Results ---")
    print(f"Mass: {M:.2e} kg (~{M / M_sun:.2e} solar masses)")
    print(f"Hawking Temperature: {T:.2e} K")
    print(f"Luminosity: {L:.2e} W")
    print(f"Evaporation Time: {t_evap_years:.2e} years")
    
    # Plot mass decay
    times, masses = mass_decay(M)
    plt.figure(figsize=(10, 6))
    plt.loglog(times, masses, 'r-', linewidth=2)
    plt.title("Black Hole Mass Decay Due to Hawking Radiation")
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.grid(True, which="both", ls="--")
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:




