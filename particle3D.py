"""
A class to describe point particles in 3D space

An instance describes a particle in Euclidean 3D space: 
velocity and position are [3] arrays

"""
import numpy as np

class Particle3D(object):
    """
    Class to describe point-particles in 3D space

    Attributes
    ----------
    label: name of the particle
    mass: mass of the particle
    position: position of the particle
    velocity: velocity of the particle

    Methods
    -------
    __init__
    __str__
    kinetic_energy: computes the kinetic energy
    momentum: computes the linear momentum
    update_position_1st: updates the position to 1st order
    update_position_2nd: updates the position to 2nd order
    update_velocity: updates the velocity

    Static Methods
    --------------
    read_file: initializes a P3D instance from a file handle
    total_kinetic_energy: computes total K.E. of a list of particles
    com_velocity: computes centre-of-mass velocity of a list of particles
    """

    def __init__(self, label, mass, position, velocity):
        """
        Initialises a particle in 3D space.

        Parameters
        ----------
        label: str
            name of the particle
        mass: float
            mass of the particle
        position: [3] float array
            position vector
        velocity: [3] float array
            velocity vector
        """
        #initialise particle properties, converting inputs to correct types: str/float/numpy array
        self.label = str(label) 
        self.mass = float(mass)
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __str__(self):
        """
        Return an XYZ-format string. The format is
        label    x  y  z

        Returns
        -------
        str
        """
        #return the string in the above format, by accessing each element of positon array
        return (self.label + " " + str(self.position[0]) + " " + str(self.position[1]) + " " + str(self.position[2]))
    
    def kinetic_energy(self):
        """
        Returns the kinetic energy of a Particle3D instance

        Returns
        -------
        ke: float
            1/2 m v**2
        """
        #calculate total kinetic energy of P3D instance using the magnitude of velocity**2
        ke = (0.5*self.mass*(np.linalg.norm(self.velocity))**2)
        
        return ke

    def momentum(self):
        """
        Returns the linear momentum of a Particle3D instance
        
        Returns
        -------
        p: array
           m*v
        """
        #calculate momentum of P3D instance
        p = self.mass*self.velocity
        
        return  p

    def update_position_1st(self, dt):
        """
        Returns the 1st order position update
        
        Returns
        -------
        self.position: array
        """
        #update the position to the 1st order
        self.position = self.position + dt*self.velocity
        
        return self.position
        
        
    def update_position_2nd(self, dt, force):
        """
        Returns the 2nd order position update
        
        Returns
        -------
        self.position: array
        """
        #update the position to the 2nd order
        self.position = self.position + dt*self.velocity + ((dt**2)*((force)/(2*self.mass)))
        
        return self.position

    def update_velocity(self, dt, force):
        """
        Returns the updated velocity
        
        Returns
        -------
        self.velocity: array
        """
        #update the velocity
        self.velocity = self.velocity + dt*((force)/(self.mass))
        
        return self.velocity

    @staticmethod
    def read_line(line):
        """
        Creates a Particle3D instance given a line of text.

        The input line should be in the format:
        label   <mass>  <x> <y> <z>    <vx> <vy> <vz>

        Parameters
        ----------
        filename: str
            Readable file handle in the above format

        Returns
        -------
        p: Particle3D
        """
        filename = line.split() #split input line (str) into a list where each particle attribute is an element of the list
        label = filename[0] #assign each element to its particle attribute in the code, converting to the correct type (str/float/numpy array)
        mass = float(filename[1])
        position = np.array([float(filename[2]), float(filename[3]), float(filename[4])])
        velocity = np.array([float(filename[5]), float(filename[6]), float(filename[7])])
        p = Particle3D(label, mass, position, velocity) #create a Particle3D object from given attributes
        
        return p

    @staticmethod
    def total_kinetic_energy(particles):
        """
        Returns the total kinetic energy of a list of particles
        
        Returns
        -------
        total_ke: float
        """
        total_ke = 0 #start kinetic energy counter value at 0
        for p in particles:
            total_ke += p.kinetic_energy() #loop over list of particles summing kinetic energy of each particle
        
        return total_ke
        

    @staticmethod
    def com_velocity(particles):
        """
        Computes the CoM velocity of a list of P3D's

        Parameters
        ----------
        particles: list
            A list of Particle3D instances

        Returns
        -------
        com_vel: array
        """
        total_mass = 0 #start mass and momentum counter values at 0
        total_momentum = 0
        for p in particles:
            total_mass += p.mass() #loop over list of particles summing mass of each particle 
            total_momentum += p.momentum() #loop over list of particles summing momentum of each particle 
            
            
        com_vel = total_momentum/total_mass #calculate the CoM velocity from updated counter values
        
        return com_vel
      
