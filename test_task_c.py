import numpy as np


def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    eps = 1e-10
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    dq = q / n_phi
    
    x_charge = a * np.cos(phi)
    y_charge = a * np.sin(phi)
    z_charge = 0.0
    
    dx = x - x_charge
    dy = y - y_charge
    dz = z - z_charge
    
    r = np.sqrt(dx**2 + dy**2 + dz**2 + eps)
    dV = dq / r
    
    V = np.sum(dV)
    return V


def ring_potential_grid(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720):
    eps = 1e-10
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dq = q / n_phi
    
    phi = phi.reshape(-1, 1, 1)
    y = np.asarray(y_grid).reshape(1, 1, -1)
    z = np.asarray(z_grid).reshape(1, -1, 1)
    
    x_charge = a * np.cos(phi)
    y_charge = a * np.sin(phi).reshape(-1, 1, 1)
    
    dx = x0 - x_charge
    dy = y - y_charge
    dz = z
    
    r = np.sqrt(dx**2 + dy**2 + dz**2 + eps)
    dV = dq / r
    
    V = np.sum(dV, axis=0)
    return V


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    return q / np.sqrt(a * a + z * z)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    a, q = 1.0, 1.0
    
    plt.figure(figsize=(8, 6))
    ys = np.linspace(-3, 3, 100)
    zs = np.linspace(-3, 3, 100)
    V = ring_potential_grid(ys, zs, x0=0, a=a, q=q, n_phi=720)
    im = plt.pcolormesh(ys, zs, V, cmap='viridis', shading='auto')
    
    phi_circle = np.linspace(0, 2*np.pi, 100)
    plt.plot(a*np.cos(phi_circle), a*np.sin(phi_circle), 'r--', linewidth=2, label='Ring position')
    plt.colorbar(im, label='Potential V')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Potential of Charged Ring (y-z plane)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('ring_potential_heatmap.png', dpi=150)
    
    plt.figure(figsize=(8, 5))
    z_axis = np.linspace(0, 5, 50)
    V_num = np.array([ring_potential_point(0, 0, z, a=a, q=q, n_phi=720) for z in z_axis])
    V_ana = axis_potential_analytic(z_axis, a=a, q=q)
    plt.plot(z_axis, V_num, 'o', markersize=4, label='Numerical')
    plt.plot(z_axis, V_ana, '-', linewidth=2, label='Analytic')
    plt.xlabel('z')
    plt.ylabel('Potential V')
    plt.title('Potential on z-axis: Numerical vs Analytic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ring_axis_comparison.png', dpi=150)
    
    plt.show()
