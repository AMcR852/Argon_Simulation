"""
Project: auxiliary MD methods
"""
import math
import numpy as np
from particle3D import Particle3D

def generate_initial_conditions(N, rho, T):
    """
    Parameters
    ----------
    N : float
        Number of particles
    rho : float
        The density in reduced units.
    T : float
        The temperature in reduced units.
    Returns
    -------
    None.
    
    """
    # initialise list of particles with zero position and velocity, and label
    # equal to their number in the list
    particles = []
    for i in range(1, N+1):
        particles.append(Particle3D((str(i)), 1, np.array([0,0,0]), np.array([0,0,0])))
    # set initial positions, velocities and box size
    box_size = set_initial_positions(rho, particles)[0] 
    set_initial_velocities(T, particles) 
    
    return particles, box_size

def set_initial_positions(rho, particles):
    """
    Set the positions of a list of particles following
    a face-centered cubic lattice structure with a given density,
    and compute the size of the cube.

    The cube does not have to be completely filled, but a note
    will be printed if it is not. It is filled when the number
    of particles is 4 n ** 3 for some n.

    Parameters
    ----------
    rho: float
        The density in reduced units.
    particle: list
        A list of Particle3D objects.
    """
    # Determine number of particles
    natoms = len(particles)
    
    # Set box dimensions
    box_size = (natoms/rho)**(1./3.)
    
    # Number or particles in each direction
    ndim = math.ceil( (natoms/4.0)**(1./3.) )
    
    # Give warning if fcc lattice will not be fully occupied
    full_lattice = True
    if 4*ndim**3 != natoms:
        full_lattice = False
        print("   Atoms will not fill a fcc lattice completely.")
        print("   Occupancy factor = {0:5.3f}".format(natoms/4*ndim**3))
            
    # Separation between particles
    delta = box_size / ndim
    print("The nearest-neighbour distance is {0:6.3f} [L]\n".format(delta*np.sqrt(2)/2))
    
    # Set particle positions
    fcc = 0.5*np.array([[0, 0, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]],float)

    all_sites = np.zeros([4*ndim**3, 3])

    n_lattice = 0
    for i in range(ndim):
        for j in range(ndim):
            for k in range(ndim):
                all_sites[n_lattice:n_lattice+4] = np.array([i,j,k]) + fcc
                n_lattice += 4
    all_sites *= delta

    for particle, position in zip(particles,all_sites):
        particle.position = position

    # Some output
    print("{0:d} atoms placed on a face-centered cubic lattice.\n".format(natoms))
    print("Box dimensions: {0:f} {0:f} {0:f}\n".format(box_size))
    
    # Return the box size as Numpy array, and whether the lattice is fully occupied
    return box_size*np.ones(3), full_lattice
    

def set_initial_velocities(T, particles, seed=None):
    """
    Set the velocities of a list of particles at random so they
    have the correct total kinetic energy for a given temperature.

    Parameters
    ----------
    temperature: float
        The temperature in reduced units.
    particles: list
        A list of Particle3D objects.
    seed: int
        (Optional) choose a random seed to always get the same
        velocities
    """
    # Set the random seed to ease up debugging. Choosing
    # and integer seed makes 
    try:
        prng = np.random.default_rng(seed)
    except:
        print("WARNING, {0} is not a valid random seed\n".format(seed))
        print("Continuing without setting the seed...\n")
        prng = np.random.default_rng()

    # Check the temperature can be a valid, positive float
    # NOTE: This should be done at parameter read
    try:
        T = float(T)
    except:
        raise ValueError(f"Could not convert temperature {T} to a valid float")
    if T < 0.0:
        raise ValueError(f"T must be positive, not {T}")

    # Initialisation
    natoms = len(particles)
    velocities = prng.random([natoms,3])

    # Ensure CoM does not move
    v_com = np.sum(velocities, axis=0) / natoms
    velocities -= v_com
    # Re-compute final CoM - this should be close to zero.
    v_com = np.sum(velocities, axis=0) / natoms

    # Rescale velocities so that total e_kin = 3/2 N_atom kB T. Assumes m=1
    e_kin = np.sum(velocities ** 2)
    velocities *= math.sqrt(3 * natoms * T / e_kin)

    # Assign each velocity to a particle
    for particle, velocity in zip(particles, velocities):
        particle.velocity = velocity

    # Output as a sanity check
    kinetic = 0.5*np.sum(velocities**2)
    print(f"T = {T}     Total Kinetic Energy = {kinetic}")
    print(f"Centre-of-mass velocity = {v_com}\n")
    

def basic_test():
    """
    This function sets up two particles which should oscillate
    under the Lennard-Jones force if your integration is working
    correctly.

    You can use it like this:
        particles, box_size = basic_test()

    Parameters
    ----------
    None

    Returns
    -------
    particles: list of Particle3D objects
        A list of particles to use in a basic test
    box_size: float
        The size of box to use in a basic test
    """
    mass = 1
    # r0 is the equilibrium separation, plus a small difference
    r0 = 2 ** (1 / 6) + 0.1
    # x0 is an offset applied to both particles
    x0 = 1.0
    x1 = np.array([x0, 0.0, 0.0])
    x2 = np.array([x0 + r0, 0.0, 0.0])
    # The particles travel in opposite directions, towards each other
    v1 = np.array([ 0.01, 0.0, 0.0])
    v2 = np.array([-0.01, 0.0, 0.0])
    particles = [
        Particle3D(f'p_0', mass, x1, v1),
        Particle3D(f'p_1', mass, x2, v2),
    ]
    box_size = 3.0
    return particles, box_size
