import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    """
    Calculate electric potential at a point (x, y, z) due to a uniformly charged ring.
    
    Parameters:
        x, y, z: Coordinates of the point
        a: Radius of the ring
        q: Charge parameter (total charge Q = 4πε₀q)
        n_phi: Number of discrete points for integration
        
    Returns:
        Electric potential at the point
    """
    # Generate phi angles
    phi = np.linspace(0, 2 * np.pi, n_phi)
    
    # Calculate distance from each ring element to the point
    dx = x - a * np.cos(phi)
    dy = y - a * np.sin(phi)
    dz = z * np.ones_like(phi)
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Avoid division by zero
    r = np.maximum(r, 1e-12)
    
    # Discrete integral using trapezoidal rule
    dphi = 2 * np.pi / n_phi
    integral = np.sum(1.0 / r) * dphi
    
    # Potential formula
    potential = q / (2 * np.pi) * integral
    
    return potential


def ring_potential_grid(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720):
    """
    Calculate electric potential on a y-z grid at fixed x = x0.
    
    Parameters:
        y_grid: 1D array of y coordinates
        z_grid: 1D array of z coordinates
        x0: Fixed x coordinate
        a: Radius of the ring
        q: Charge parameter
        n_phi: Number of discrete points for integration
        
    Returns:
        2D array of potential values
    """
    # Create meshgrid
    Y, Z = np.meshgrid(y_grid, z_grid)
    
    # Initialize potential matrix
    potential = np.zeros_like(Y)
    
    # Calculate potential at each grid point
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            potential[i, j] = ring_potential_point(x0, Y[i, j], Z[i, j], a, q, n_phi)
    
    return potential


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    Analytic potential along the z-axis (x=0, y=0).
    
    Parameters:
        z: z-coordinate
        a: Radius of the ring
        q: Charge parameter
        
    Returns:
        Electric potential at the point
    """
    return q / np.sqrt(a * a + z * z)


def calculate_electric_field(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720, h: float = 1e-6):
    """
    Calculate electric field using finite difference of potential.
    
    Parameters:
        y_grid: 1D array of y coordinates
        z_grid: 1D array of z coordinates
        x0: Fixed x coordinate
        a: Radius of the ring
        q: Charge parameter
        n_phi: Number of discrete points for integration
        h: Step size for finite difference
        
    Returns:
        Ey, Ez: Electric field components
    """
    # Calculate potential grid
    potential = ring_potential_grid(y_grid, z_grid, x0, a, q, n_phi)
    
    # Calculate grid spacing
    dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else h
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else h
    
    # Calculate electric field components using central difference
    Ey, Ez = np.gradient(-potential, dy, dz)
    
    return Ey, Ez


def visualize_potential_and_field():
    """
    Visualize potential and electric field in the y-z plane.
    """
    # Define grid
    y = np.linspace(-3, 3, 100)
    z = np.linspace(-3, 3, 100)
    
    # Calculate potential
    potential = ring_potential_grid(y, z, x0=0.0)
    
    # Calculate electric field
    Ey, Ez = calculate_electric_field(y, z, x0=0.0)
    
    # Calculate field magnitude
    E_mag = np.sqrt(Ey**2 + Ez**2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Potential contour
    contour_levels = np.linspace(0.2, 1.0, 20)
    im1 = ax1.contourf(y, z, potential, levels=contour_levels, cmap='viridis')
    ax1.contour(y, z, potential, levels=contour_levels, colors='white', linewidths=0.5)
    ax1.set_title('Electric Potential of Charged Ring (yz-plane)')
    ax1.set_xlabel('y')
    ax1.set_ylabel('z')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Potential')
    
    # Plot 2: Electric field
    # Downsample for quiver plot
    skip = 5
    y_skip = y[::skip]
    z_skip = z[::skip]
    Ey_skip = Ey[::skip, ::skip]
    Ez_skip = Ez[::skip, ::skip]
    E_mag_skip = E_mag[::skip, ::skip]
    
    # Normalize field vectors for visualization
    E_norm = np.sqrt(Ey_skip**2 + Ez_skip**2)
    E_norm = np.maximum(E_norm, 1e-12)  # Avoid division by zero
    Ey_normalized = Ey_skip / E_norm
    Ez_normalized = Ez_skip / E_norm
    
    im2 = ax2.contourf(y, z, E_mag, cmap='plasma')
    ax2.quiver(y_skip, z_skip, Ey_normalized, Ez_normalized, color='white', scale=50)
    ax2.set_title('Electric Field of Charged Ring (yz-plane)')
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Field Magnitude')
    
    plt.tight_layout()
    plt.savefig('ring_potential_field.png', dpi=150)
    print("Potential and field visualization saved as 'ring_potential_field.png'")


def verify_analytic_solution():
    """
    Verify the numerical solution against the analytic solution along the z-axis.
    """
    # Test points along z-axis
    z_values = np.linspace(-3, 3, 50)
    
    # Calculate numerical and analytic potentials
    numerical_potential = []
    analytic_potential = []
    
    for z in z_values:
        numerical = ring_potential_point(0.0, 0.0, z)
        analytic = axis_potential_analytic(z)
        numerical_potential.append(numerical)
        analytic_potential.append(analytic)
    
    # Calculate error
    numerical_potential = np.array(numerical_potential)
    analytic_potential = np.array(analytic_potential)
    error = np.abs(numerical_potential - analytic_potential)
    max_error = np.max(error)
    
    print(f"Maximum error along z-axis: {max_error:.2e}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, numerical_potential, 'b-', label='Numerical')
    plt.plot(z_values, analytic_potential, 'r--', label='Analytic')
    plt.xlabel('z')
    plt.ylabel('Potential')
    plt.title('Comparison of Numerical and Analytic Potentials along z-axis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ring_potential_verification.png', dpi=150)
    print("Verification plot saved as 'ring_potential_verification.png'")


if __name__ == "__main__":
    # Verify analytic solution
    verify_analytic_solution()
    
    # Visualize potential and field
    visualize_potential_and_field()
    
    # Test some specific points
    print("\nPotential at specific points:")
    print(f"Center (0,0,0): {ring_potential_point(0, 0, 0):.4f}")
    print(f"Along z-axis (0,0,1): {ring_potential_point(0, 0, 1):.4f} (analytic: {axis_potential_analytic(1):.4f})")
    print(f"Along y-axis (0,2,0): {ring_potential_point(0, 2, 0):.4f}")

