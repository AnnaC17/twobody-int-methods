import numpy as np

##########

def get_accel(X,x,G=1.0,M=1.0,m=3.0e-6):
    '''
    PURPOSE:
            Calculates the acceleration of both bodies from a set of initial conditions for
            a given length of time

    PARAMETERS:
            X; np.array:  position of body 1
            x; np.array:  position of body 2
            
    OPTIONAL PARAMETERS:
            M; float:     mass of body 1 (default = 1.0)
            m; float:     mass of body 2 (default = 3.0e-6)
            G; float:     gravitational constant (default = 1.0)

    RETURNS:
            A; float:     acceleration of body 1
            a; float:     acceleration of body 2
    '''
    
    # Acceleration due to gravity
    a = -G*M*(x-X)/(np.linalg.norm(x-X))**3
    A = -G*m*(X-x)/(np.linalg.norm(X-x))**3
    
    return a,A

##########

def get_energy(X,x,V,v,G=1.0,M=1.0,m=3.0e-6):
    '''
    PURPOSE:
            Calculates the energy of the entire system from a set of conditions
            
    PARAMETERS:
            X; np.array:  position of body 1
            x; np.array:  position of body 2
            V; np.array:  velocity of body 1
            v; np.array:  velocity of body 2
            
    OPTIONAL PARAMETERS:
            M; float:     mass of body 1 (default = 1.0)
            m; float:     mass of body 2 (default = 3.0e-6)
            G; float:     gravitational constant (default = 1.0)
            
    RETURNS:
            E; float:     the energy of the system
    '''
    
    
    # Magnitudes of the position vector between body 1 & 2, and velocity vectors
    r_mag = np.linalg.norm(x-X)
    V_mag = np.linalg.norm(V)
    v_mag = np.linalg.norm(v) 
    
    # Defining gravitational potential energy & kinetic energy
    potential = -G*M*m/r_mag
    kinetic = 0.5*m*v_mag**2 + 0.5*M*V_mag**2
    
    # Total energy given by the sum
    E = potential + kinetic
    
    return E
    
##########

def euler(X,x,V,v,dt,t_end,M=1.0,m=3.0e-6,G=1.0):
    '''
    PURPOSE:
            Calculates the orbits of the two bodies using the Euler Method.
            
    PARAMETERS:
            X; np.array:  position of body 1
            x; np.array:  position of body 2
            V; np.array:  velocity of body 1
            v; np.array:  velocity of body 2
            dt; int:      timestep
            t_end; int:   number of timesteps
            
    OPTIONAL PARAMETERS:
            M; float:     mass of body 1 (default = 1.0)
            m; float:     mass of body 2 (default = 3.0e-6)
            G; float:     gravitational constant (default = 1.0)
            
    RETURNS:
            X_vec; list:  array of body 1's x position
            Y_vec; list:  array of body 1's y position
            x_vec; list:  array of body 2's x position
            y_vec; list:  array of body 2's y position
            E; list:      array of recorded energies
    '''
    x_vec, y_vec = [],[]
    X_vec, Y_vec = [],[]
    E = []
    
    for i in np.arange(0,t_end,dt):
        
        # grab accel
        a, A = get_accel(X,x)
        
        # kick position
        x = x + v*dt
        X = X + V*dt
        
        # kick velocity
        v = v + a*dt
        V = V + A*dt
        
        # get energy
        energy = get_energy(X,x,V,v)
        
        # store values
        x_vec.append(x[0]), y_vec.append(x[1])
        X_vec.append(X[0]), Y_vec.append(X[1])
        E.append(energy)
        
    return X_vec, Y_vec, x_vec, y_vec, E
    
##########

def RK4(X,x,V,v,dt,t_end,M=1.0,m=3.0e-6,G=1.0):
    '''
    PURPOSE:
            Calculates the orbits of the two bodies to fourth order using the Runge-Kutta method.
            
    PARAMETERS:
            X; np.array:  position of body 1
            x; np.array:  position of body 2
            V; np.array:  velocity of body 1
            v; np.array:  velocity of body 2
            dt; int:      timestep
            t_end; int:   number of timesteps
            
    OPTIONAL PARAMETERS:
            M; float:     mass of body 1 (default = 1.0)
            m; float:     mass of body 2 (default = 3.0e-6)
            G; float:     gravitational constant (default = 1.0)
            
    RETURNS:
            X_vec; list:  x position of body 1
            Y_vec; list:  y position of body 1
            x_vec; list:  x position of body 2
            y_vec; list:  y position of body 2
            E; list:      array of recorded energies
    '''
    x_vec, y_vec = [],[]
    X_vec, Y_vec = [],[]
    E = []
    
    for i in np.arange(0,t_end,dt):
        
        # grab acceleration
        a, A = get_accel(X,x)
        
        # using K1 values
        x1 = x + 0.5*v*dt
        X1 = X + 0.5*V*dt
        
        v1 = v + 0.5*a*dt
        V1 = V + 0.5*A*dt
        
        a1, A1 = get_accel(X1,x1)
        
        # using K2 values
        x2 = x + 0.5*v1*dt
        X2 = X + 0.5*V1*dt
        
        v2 = v + 0.5*a1*dt
        V2 = V + 0.5*A1*dt
        
        a2, A2 = get_accel(X2,x2)
        
        # using K3 values
        x3 = x + v2*dt
        X3 = X + V2*dt
        
        v3 = v + a2*dt
        V3 = V + A2*dt
        
        a3, A3 = get_accel(X3,x3)
        
        # finally, we get:

        x = x + (1/6)*(v + 2*v1 +2*v2 + v3)*dt
        X = X + (1/6)*(V + 2*V1 + 2*V2 + V3)*dt
        
        v = v + (1/6)*(a + 2*a1 + 2*a2 + a3)*dt
        V = V + (1/6)*(A + 2*A1 + 2*A2 + A3)*dt
        
        # get energy too
        energy = get_energy(X,x,V,v)
        
        # storing values now
        x_vec.append(x[0]), y_vec.append(x[1])
        X_vec.append(X[0]), Y_vec.append(X[1])
        E.append(energy)
        
    return X_vec, Y_vec, x_vec, y_vec, E

##########

def leapfrog(X,x,V,v,dt,t_end,M=1.0,m=3.0e-6,G=1.0):
    '''
    PURPOSE:
            Calculates the orbits of the two bodies using the Leapfrog/Kick-Drift-Kick Method.
            
    PARAMETERS:
            X; np.array:  position of body 1
            x; np.array:  position of body 2
            V; np.array:  velocity of body 1
            v; np.array:  velocity of body 2
            dt; int:      timestep
            t_end; int:   number of timesteps
            
    OPTIONAL PARAMETERS:
            M; float:     mass of body 1 (default = 1.0)
            m; float:     mass of body 2 (default = 3.0e-6)
            G; float:     gravitational constant (default = 1.0)
            
    RETURNS:
            X_vec; list:  x position of body 1
            Y_vec; list:  y position of body 1
            x_vec; list:  x position of body 2
            y_vec; list:  y position of body 2
    '''
    x_vec, y_vec = [],[]
    X_vec, Y_vec = [],[]
    E = []
    
    for i in np.arange(0,t_end,dt):
        
        # get the accelerations
        a, A = get_accel(X,x)
        
        # half step kick
        v2 = v + 0.5*a*dt
        V2 = V + 0.5*A*dt
        
        # drift
        x = x + v2*dt
        X = X + V2*dt
        
        # accel part 2!
        a2, A2 = get_accel(X,x)
        
        # new velocities
        v = v2 + 0.5*a2*dt
        V = V2 + 0.5*A2*dt
        
        # get energy
        energy = get_energy(X,x,V,v)
        
        x_vec.append(x[0]), y_vec.append(x[1])
        X_vec.append(X[0]), Y_vec.append(X[1])
        E.append(energy)
        
    return x_vec, y_vec, X_vec, Y_vec, E
