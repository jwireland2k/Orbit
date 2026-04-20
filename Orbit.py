import autograd.numpy as np
from autograd import grad
import pygame

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

D = 2 # dimensionality of space (2D, 3D etc.)
N = 4 # number of bodies - currently Sun, Earth, Mars - we will increase this later


# Masses in solar mass units
m_s = 1.0
m_e = 0.000003 
m_m = 0.00000032
m_o = 0.005

m = [m_s, m_e, m_m, m_o] # pack masses into a 1D array

M = np.repeat(m, D)  # repeat masses for each dimension

# Initial position vectors in X,Y coordinates
r_s = [0.0, 0.0]
r_e = [0.1, -1.0] # In astronomical units
r_m = [0.0, -1.524] # In astronomical units
r_o = [0.1, -2.0]

# Initial velocity vectors in X,Y coordinates
v_s = [0.0, 0.0] # In astronomical units per year
v_e = [-6.32, 0.0] # In astronomical units per year
v_m = [-5.05, 0.0] # In astronomical units per year
v_o = [4, 1.0]

# pack into NxD array and vector log length N*D
r0 = np.reshape([r_s, r_e, r_m, r_o], [N, D])
R0 = np.reshape(r0, N*D) # pack into 1D array

# pack into NxD array and vector log length N*D
v0 = np.reshape([v_s, v_e, v_m, v_o], [N, D])
V0 = np.reshape(v0, N*D) # pack into 1D array

G = 37.95  # Gravitational constant

# Simulation time variables
dt = 0.001
t = np.arange(0.0, 2, dt)
T = len(t) # Total simulation points


def potential_energy(R):
    r = np.reshape(R, (N, D)) # convert from 1d vector back to D-dimensional (here 2D)
    U = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = np.linalg.norm(r[i] - r[j]) # distance from i to j
                U += -G/2.0 * m[i]*m[j] / r_ij # contribution to potential
    return U


grad_potential_energy = grad(potential_energy)
UR = grad_potential_energy(R0)
h = 1e-5
dU = np.zeros(N*D)

for i in range(N*D):
    Rp = R0.copy()
    Rp[i] += h
    dU[i] = (potential_energy(Rp) - potential_energy(R0)) / h


def forward_euler(R0, V0):
    R = R0.copy()        # current position
    V = V0.copy()        # current velocity
    V_i = V.copy()
    A = -grad_potential_energy(R) / M # acceleration from Newton's 2nd law (F = ma)
    V += A * dt     # Update velocities
    R += V_i * dt   # Update positions
    return (R, V)


def symplectic_euler(R0, V0):
    R = R0.copy()        # current position
    V = V0.copy()        # current velocity
    A = -grad_potential_energy(R) / M  # acceleration from Newton's 2nd law (F = ma)
    V += A * dt      # Update velocities
    R += V * dt      # Update positions
    return (R, V)


pygame.init()
infos = pygame.display.Info()
screen_width, screen_height = (infos.current_w, infos.current_h)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
running = True

def plot_planets(planetPos, colourOffset):
    pygame.draw.circle(screen, (255,255,colourOffset), (screen_width/2+int(planetPos[0]*300), int(screen_height/2+planetPos[1]*300)), 20)
    pygame.draw.circle(screen, (colourOffset, 0, 255), (screen_width/2+int(planetPos[2]*300), int(screen_height/2+planetPos[3]*300)), 5)
    pygame.draw.circle(screen, (255, 0, colourOffset), (screen_width/2+int(planetPos[4]*300), int(screen_height/2+planetPos[5]*300)), 4)
    pygame.draw.circle(screen, (255, colourOffset, 0), (screen_width/2+int(planetPos[6]*300), int(screen_height/2+planetPos[7]*300)), 4)


R1, V1 = R0, V0
R2, V2 = R0, V0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    screen.fill((0,0,0))
    #forward euler
    #R1, V1 = forward_euler(R1, V1)
    #plot_planets(R1, 0)

    #symplectic euler
    R2, V2 = symplectic_euler(R2, V2)
    plot_planets(R2, 100)

    pygame.display.flip()
    pygame.time.wait(50)
