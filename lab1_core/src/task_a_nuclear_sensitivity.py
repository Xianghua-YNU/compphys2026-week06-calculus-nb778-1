import numpy as np


def rate_3alpha(T: float) -> float:
    """
    Calculate 3-alpha reaction rate.
    
    Parameters:
        T: Temperature in Kelvin
        
    Returns:
        Reaction rate q(T)
    """
    T8 = T / 1.0e8
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)


def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    """
    Calculate dq/dT using forward difference.
    
    Parameters:
        T0: Reference temperature
        h: Step size factor
        
    Returns:
        Approximation of dq/dT at T0
    """
    # Calculate delta T as h * T0
    delta_T = h * T0
    
    # Forward difference formula
    q_T0 = rate_3alpha(T0)
    q_T0_plus = rate_3alpha(T0 + delta_T)
    
    return (q_T0_plus - q_T0) / delta_T


def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    """
    Calculate temperature sensitivity index nu.
    
    Parameters:
        T0: Reference temperature
        h: Step size factor
        
    Returns:
        Temperature sensitivity index nu
    """
    q_T0 = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    
    return (T0 / q_T0) * dq_dT


def nu_table(T_values, h: float = 1e-8):
    """
    Generate table of temperature and corresponding nu values.
    
    Parameters:
        T_values: List of temperatures
        h: Step size factor
        
    Returns:
        List of tuples (T, nu(T))
    """
    result = []
    for T in T_values:
        if T <= 0:
            raise ValueError("Temperature must be positive")
        nu = sensitivity_nu(T, h)
        result.append((T, nu))
    return result


if __name__ == "__main__":
    # Test temperatures
    test_temperatures = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    
    print("Temperature Sensitivity Index Table")
    print("=" * 40)
    
    # Generate and print the table
    table = nu_table(test_temperatures)
    for T, nu in table:
        print(f"{T:.3e} K : nu = {nu:.2f}")
    
    print("=" * 40)
