import math


def debye_integrand(x: float) -> float:
    """
    Debye integrand function.
    
    Parameters:
        x: Integration variable
        
    Returns:
        Value of the integrand
    """
    if abs(x) < 1e-10:
        # Taylor expansion for small x: x^4 * e^x / (e^x - 1)^2 -> x^2
        return x * x
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    """
    Composite trapezoidal rule for numerical integration.
    
    Parameters:
        f: Integrand function
        a: Lower limit
        b: Upper limit
        n: Number of intervals
        
    Returns:
        Approximate integral value
    """
    if n < 1:
        raise ValueError("Number of intervals must be at least 1")
    
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)
    
    integral *= h
    return integral


def simpson_composite(f, a: float, b: float, n: int) -> float:
    """
    Composite Simpson's rule for numerical integration.
    
    Parameters:
        f: Integrand function
        a: Lower limit
        b: Upper limit
        n: Number of intervals (must be even)
        
    Returns:
        Approximate integral value
    """
    if n < 2:
        raise ValueError("Number of intervals must be at least 2")
    if n % 2 != 0:
        raise ValueError("Number of intervals must be even for Simpson's rule")
    
    h = (b - a) / n
    integral = f(a) + f(b)
    
    for i in range(1, n, 2):
        x_i = a + i * h
        integral += 4 * f(x_i)
    
    for i in range(2, n, 2):
        x_i = a + i * h
        integral += 2 * f(x_i)
    
    integral *= h / 3
    return integral


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    """
    Calculate Debye integral I(y) where y = theta_d / T.
    
    Parameters:
        T: Temperature in Kelvin
        theta_d: Debye temperature in Kelvin
        method: Integration method ('trapezoid' or 'simpson')
        n: Number of intervals
        
    Returns:
        Value of the Debye integral
    """
    if T <= 0:
        raise ValueError("Temperature must be positive")
    
    y = theta_d / T
    
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method.lower() == "simpson":
        # Ensure n is even for Simpson's rule
        if n % 2 != 0:
            n += 1
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError("Method must be 'trapezoid' or 'simpson'")


if __name__ == "__main__":
    # Test temperatures
    test_temperatures = [10, 50, 100, 300, 500, 1000]
    theta_d = 428.0
    n = 200
    
    print("Debye Integral Calculation")
    print("=" * 60)
    print(f"Debye temperature: {theta_d} K")
    print(f"Number of intervals: {n}")
    print("=" * 60)
    print(f"{'Temperature (K)':<15} {'y=theta_d/T':<12} {'Trapezoid':<12} {'Simpson':<12} {'Difference':<12}")
    print("-" * 60)
    
    for T in test_temperatures:
        y = theta_d / T
        
        # Calculate using both methods
        integral_trap = debye_integral(T, theta_d, method="trapezoid", n=n)
        integral_simp = debye_integral(T, theta_d, method="simpson", n=n)
        difference = abs(integral_simp - integral_trap)
        
        print(f"{T:<15} {y:<12.4f} {integral_trap:<12.6f} {integral_simp:<12.6f} {difference:<12.2e}")
    
    print("=" * 60)
