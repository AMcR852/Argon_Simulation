"""
Project: Full Source Code

N-Body simulation for solid, liquid and gas phases of Argon using period boundary conditions.

"""
import sys
import numpy as np
from particle3D import Particle3D
from lj_utils import generate_initial_conditions
import matplotlib.pyplot as pyplot

def user_input():
    """
    Gets the simulation parameters from the input file defined on the command line.
    Gets the name of output file defined on the command line.
    
    Returns
    -------
    parameters : list of simulation parameters
    VMDfile : name of output file for plotting to VMD
    """
    
    if len(sys.argv) != 3:
        print("Wrong number of arguments given. Provide 3.")
       
    parameterfile = sys.argv[1]
    VMDfile = sys.argv[2]
    parameterfile = open(parameterfile, 'r')
    lines = parameterfile.readlines()
    N = int(lines[1])
    rho = float(lines[3])
    T = float(lines[5])
    dt = float(lines[7])
    numstep = int(lines[9])
    parameters = [N, rho, T, dt, numstep]

    return parameters, VMDfile

def get_parameters():
    
    # set simulation parameters
    parameters, VMDfile = user_input()
    N = parameters[0]
    rho = parameters[1]
    T = parameters[2]
    dt = parameters[3]
    numstep = parameters[4]

    # set initial conditions
    particles, box_size = generate_initial_conditions(N, rho, T)

def compute_separations(particles, box_size):
    """
    Computes the vector separations, using the MIC, between all the pairs of particles 
    
    Parameters
    ----------
    particles : list of Particle3D objects
    box_size : float. Dimension of box for periodic boundary conditions 

    Returns
    -------
    separations : (n,n,3) numpy array where n is the number of particles
    """
    n = len(particles) # get number of particles 
    separations = np.zeros((n,n,3)) 
    for i in range(1, n): # skip first element since it will be zero
        for j in range(i): 
            distance = particles[i].position - particles[j].position 
            rem = np.mod(distance, box_size) # image in nearest cube
            separations[i][j] = -1*np.array(np.mod(rem + 0.5*box_size, box_size) - 0.5*box_size) # get the MIC separation vector for each particle
            separations[j][i] = -1*separations[i][j] # avoid calculating MIC vectors twice by reflecting matrix about diagonal
    
    return separations 

def compute_forces_potential(particles, separations):
    """
    Computes the Lennard-Jones forces on the particles and the total potential energy of the system
    
    Parameters
    ----------
    particles : list of Particle3D objects
    separations : (n,n,3) numpy array where n is the number of particles

    Returns
    -------
    forces : (n,3) numpy array containing the vector force element (i,j)
    potential :  float. The total potential energy of the system
    """
    n = len(particles) # get number of particles
    forces = np.zeros((n,3)) 
    potential = 0  
    
    for i in range(n):
       for j in range(i): # avoid particle double counting
          singleSeparation = separations[i][j]
          r = np.linalg.norm(singleSeparation) # get magnitude of each individual separation
          if r == 0: 
              continue # keep the loop going even if we get a zero value for denominator
          LJ_force = 48*(r**-14 - 0.5*r**-8)*singleSeparation 
          LJ_potential = 4*(r**-12 - r**-6)
          forces[j] += LJ_force
          forces[i] += -1*LJ_force # using Newtons 3rd law
          potential += LJ_potential
          
    return forces, potential

def apply_pbc(particles):
    """
    Applies periodic boundary conditions to all particles. Moves any particle that
    has gone outside the box back inside consistent with periodic boundary conditions.
    
    Parameters
    ----------
    particles : list of Particle3D objects

    Returns
    -------
    None.
    """

    for p in particles:
        p.position = np.mod(p.position, box_size)
    
    return None
   
def update_velocities(particles, forces, dt):
    """
    Conducts a 1st order velocity update on all particles
    
    Parameters
    ----------
    particles : list of Particle3D objects
    forces : (n,3) numpy array containing the vector force element (i,j)
    dt : float. Timestep increment 

    Returns
    -------
    None.
    """
    for p, force in zip(particles, forces):
        p.update_velocity(dt, force)
    
    return None
    
def update_positions(particles, forces, dt):
    """
    Conducts a 2nd order position update on all particles
    
    Parameters
    ----------
    particles : list of Particle3D objects
    forces : (n,3) numpy array containing the vector force element (i,j)
    dt : float. Timestep increment

    Returns
    -------
    None.
    """
    for p, force in zip(particles, forces):
        p.update_position_2nd(dt, force)
   
    return None
            
def get_positions(particles):
    """
    Generates [N,3] array of positions of all particles
    
    Parameters
    ----------
    particles : list of particle3D objects

    Returns
    -------
    numpy array: positions of all bodies
    """
    positions = np.array([p.position for p in particles])
    
    return positions

def get_velocities(particles):
    """
    Generates [N,3] array of velocities of all particles
    
    Parameters
    ----------
    particles : list of particle3D objects

    Returns
    -------
    numpy array: velocities of all bodies
    """
    
    velocities = np.array([p.velocity for p in particles])
    
    return velocities

def kinetic(velocities):
    """
    Calculates the total kinetic energy of the system
    
    Parameters
    ----------
    velocities : (N,3) array of system velocities

    Returns
    -------
    Total kinetic energy of the system
    """
    n = len(velocities)
    vsquared = 0
    for i in range(n):
        vsquared += np.linalg.norm(velocities[i])**2
    mass = 1

    return 0.5*mass*vsquared

def total_energy(particles, separations):
    """
    Calculates the total energy of the system (kinetic + potential)
    
    Parameters
    ----------
    particles : list of particle3D objects
    separations : (n,n,3) numpy array where n is the number of particles
    
    Returns
    -------
    A [1,3]-dim array of the kinetic, potential, and total energy at time t
    """
    
    pot_e = compute_forces_potential(particles, separations)[1]
    kin_e = kinetic(get_velocities(particles))
    total_e = pot_e + kin_e
    
    return np.array([pot_e, kin_e, total_e])
    
def VMD_data(time):
    """
    Generates an string in the XYZ format containing data to plot to VMD. It gives the position of each point
    at each time. Uses the Particle3D ___str___ method.
    
    Parameters
    ----------
    time : float. Equal to t*dt

    Returns
    -------
    VMD_output_string: string in the XYZ format for plotting in VMD
    """
    
    VMD_output_string = '%i\nPoint = %i\n' % (len(particles), time)
    for p in particles:
            VMD_output_string += "s"+str(p) + '\n'
    
    return VMD_output_string
    
def MSD(positions, box_size, numstep):
    """
    Takes in the array of particle positions and box size and calculates the mean squared displacement
    at each time value. Also takes in value of number of timesteps.
    
    Parameters
    ----------
    positions: [N,3] position array of all particles at all times
    start : start range of positins to calculate MSD
    numstep: float. The end value to sum the MSD to

    Returns
    -------
    mean_square_displacement : list.  MSD evaluated at each time
    """
    
    r_0 = positions[0]
    mean_square_displacement = []
    for t in range(numstep):
        r_t = positions[t]
        total_sum = 0
        for i in range(len(r_t)):
            distance = r_t - r_0
            rem = np.mod(distance,box_size) # image in the first cube
            mic_separation_vector = np.mod(rem+box_size/2,box_size)-box_size/2 
            total_sum += np.linalg.norm(mic_separation_vector)**2

        mean_square_displacement.append(total_sum/numstep)
        
    return mean_square_displacement

def main():
    """
    The main function:
        
    - Conducts the velocity verlet time integration scheme on all particles, using periodic
      boundary conditions
    - Calculates energy across the simulation
    - Generates numpy arrays of position & velocity across simulation
    - Saves lists of time, and all observables in the simulation
    - Plots the MSD and energy graphs
    - Saves files of energies and MSD at each time value, and VMD output file.

    Returns
    -------
    positions : [numstep, N, 3]-dim numpy array with positions for all timesteps
    time_list : [N]-dim numpy array with timestamps for each timestep
    """

    # Get parameters and set inital conditions
    get_parameters()

    # Create lists used in the simulation
    VMD_list, positions, velocities,  = [],[],[]
    KE, PE, TE = [], [], []
    time_list = []
    
    # Get initial forces and separations
    separations = compute_separations(particles, box_size)
    forces = compute_forces_potential(particles, separations)[0]
    print("Performing verlet time-integration...\n")
    
    for t in range(numstep):
        
        positions.append(get_positions(particles)) # save position array
        apply_pbc(particles) # apply periodic boundary conditions
        velocities.append(get_velocities(particles)) #save velocities array 
        time_list.append(t*dt) # save time stamp
        VMD_list.append(VMD_data(t)) # save VMD data to a list
        
        # Calculate and save energies in lists
        energies = total_energy(particles, separations)
        PE.append(energies[0])
        KE.append(energies[1])
        TE.append(energies[2])
                      
        # Update positions
        update_positions(particles, forces, dt)
        temporary_forces = forces
        separations = compute_separations(particles, box_size)
        forces = compute_forces_potential(particles, separations)[0]
        # Update velocities
        update_velocities(particles, 0.5*(temporary_forces + forces), dt)

    # Calculate MSD
    print("Calculating mean squared displacement function...\n")
    mean_square_displacement = MSD(positions, box_size, numstep)
    
    # Output VMD data to file
    vmdstring = ''.join(VMD_list) 
    with open(VMDfile, 'w') as output:
            output.write(vmdstring)
    print("Sucessful write to VMD data output file \n")
            
    # Output energy data to file
    with open("energyfile.txt", 'w') as energy:
        for i in range(len(time_list)):
            energy.write("{0}\t{timestamp:.3f}\t\n".format(TE[i], timestamp = time_list[i]))
    print("Sucessful write to energy output file \n")
    
    # Output MSD data to file
    with open('MSDfile.txt', 'w') as MSDfile:
        for i in range(len(time_list)):
            MSDfile.write("{0}\t{timestamp:.3f}\t\n".format(mean_square_displacement[i], timestamp = time_list[i]))
    print("Sucessful write to MSD output file \n")
    
    # Plot energy graphs
    print("Plotting energy graphs... \n")
    pyplot.title('Kinetic Energy vs Time')
    pyplot.xlabel('Time [t]')
    pyplot.ylabel('Kinetic Energy [\u03B5]')
    pyplot.plot(time_list, KE)
    pyplot.show()
    
    pyplot.title('Potential Energy vs Time')
    pyplot.xlabel('Time [t]')
    pyplot.ylabel('Potential Energy [\u03B5]')
    pyplot.plot(time_list, PE)
    pyplot.show()
    
    pyplot.title('Total Energy vs Time')
    pyplot.xlabel('Time [t]')
    pyplot.ylabel('Total Energy [\u03B5]')
    pyplot.plot(time_list, TE)
    pyplot.show()
    
     #Plot MSD graph
    print("Plotting mean squared displacement graph... \n")
    pyplot.title('MSD vs Time')
    pyplot.xlabel('Time [t]')
    pyplot.ylabel('MSD [\u03C3\N{SUPERSCRIPT TWO}]')
    pyplot.plot(time_list, mean_square_displacement)
    pyplot.show()
    
    # Plot combined graphs
    pyplot.figure(3)
    pyplot.plot(time_list,KE)
    pyplot.plot(time_list,PE)
    pyplot.plot(time_list,TE)
    pyplot.title("Energy vs Time")
    pyplot.xlabel("Time [t]")
    pyplot.ylabel('Energy [\u03B5]')
    pyplot.legend(['KE','PE','TE'])
    pyplot.show()
    print("Finished observable plots and writing data to output files. Simulation was sucessful.")
    return np.array(positions), np.array(time_list)

#Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
    