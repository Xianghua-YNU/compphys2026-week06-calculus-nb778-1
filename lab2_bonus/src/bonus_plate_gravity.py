import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    """
    Two-dimensional Gaussian-Legendre integration.
    
    Parameters:
        func: Integrand function f(x, y)
        ax, bx: Integration limits for x
        ay, by: Integration limits for y
        n: Number of integration points (per dimension)
        
    Returns:
        Approximate value of the double integral
    """
    # Get Gaussian-Legendre points and weights for x
    x, wx = np.polynomial.legendre.leggauss(n)
    # Transform x from [-1, 1] to [ax, bx]
    x_transformed = 0.5 * (bx - ax) * x + 0.5 * (bx + ax)
    wx_transformed = 0.5 * (bx - ax) * wx
    
    # Get Gaussian-Legendre points and weights for y
    y, wy = np.polynomial.legendre.leggauss(n)
    # Transform y from [-1, 1] to [ay, by]
    y_transformed = 0.5 * (by - ay) * y + 0.5 * (by + ay)
    wy_transformed = 0.5 * (by - ay) * wy
    
    # Compute the double integral
    integral = 0.0
    for i in range(n):
        xi = x_transformed[i]
        wxi = wx_transformed[i]
        for j in range(n):
            yj = y_transformed[j]
            wyj = wy_transformed[j]
            integral += wxi * wyj * func(xi, yj)
    
    return integral


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    """
    Calculate the gravitational force along z-axis on a particle above the center of a square plate.
    
    Parameters:
        z: Distance from the plate to the particle
        L: Side length of the square plate
        M_plate: Total mass of the plate
        m_particle: Mass of the particle
        n: Number of integration points (per dimension)
        
    Returns:
        Force along z-axis in Newtons
    """
    if z < 0:
        raise ValueError("z must be non-negative")
    
    # Surface mass density
    sigma = M_plate / (L ** 2)
    
    # Define the integrand function
    def integrand(x, y):
        r_cubed = (x**2 + y**2 + z**2) ** 1.5
        if r_cubed < 1e-12:
            return 0.0
        return z / r_cubed
    
    # Integration limits
    ax = -L/2
    bx = L/2
    ay = -L/2
    by = L/2
    
    # Compute the integral
    integral = gauss_legendre_2d(integrand, ax, bx, ay, by, n)
    
    # Calculate the force
    force = G * sigma * m_particle * integral
    
    return force


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    """
    Calculate force curve for a range of z values.
    
    Parameters:
        z_values: Array of z values
        L: Side length of the square plate
        M_plate: Total mass of the plate
        m_particle: Mass of the particle
        n: Number of integration points (per dimension)
        
    Returns:
        Array of force values
    """
    forces = []
    for z in z_values:
        force = plate_force_z(z, L, M_plate, m_particle, n)
        forces.append(force)
    return np.array(forces)


def visualize_force_curve():
    """
    Visualize the force curve as a function of z.
    """
    # Define z values from 0.2 to 10 meters
    z_values = np.linspace(0.2, 10, 100)
    
    # Calculate force curve
    forces = force_curve(z_values)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, forces, 'b-', linewidth=2)
    plt.xlabel('z (m)')
    plt.ylabel('Force Fz (N)')
    plt.title('Gravitational Force from Square Plate vs Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plate_force_curve.png', dpi=150)
    print("Force curve plot saved as 'plate_force_curve.png'")


def compare_with_point_mass():
    """
    Compare the force from the plate with that from a point mass.
    """
    # Define z values
    z_values = np.linspace(0.2, 10, 100)
    
    # Calculate force from plate
    force_plate = force_curve(z_values)
    
    # Calculate force from point mass (same total mass at origin)
    M_plate = 1.0e4  # 10 tons
    m_particle = 1.0
    force_point = G * M_plate * m_particle / (z_values ** 2)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, force_plate, 'b-', label='Square Plate')
    plt.plot(z_values, force_point, 'r--', label='Point Mass')
    plt.xlabel('z (m)')
    plt.ylabel('Force Fz (N)')
    plt.title('Comparison: Plate vs Point Mass Gravitational Force')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plate_vs_point_mass.png', dpi=150)
    print("Comparison plot saved as 'plate_vs_point_mass.png'")


def verify_near_field_limit():
    """
    Verify the near-field limit (z << L) where force should be constant.
    """
    # Small z values
    z_values = np.linspace(0.01, 1, 20)
    
    # Calculate force
    forces = force_curve(z_values)
    
    # Check if force is approximately constant
    mean_force = np.mean(forces)
    std_force = np.std(forces)
    relative_std = std_force / mean_force
    
    print(f"\nNear-field limit verification:")
    print(f"Mean force for z < 1m: {mean_force:.2e} N")
    print(f"Standard deviation: {std_force:.2e} N")
    print(f"Relative standard deviation: {relative_std:.2e}")
    print(f"Force is {'approximately constant' if relative_std < 0.01 else 'not constant'} in near-field")
