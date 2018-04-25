from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 0.1  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 0.0000 # mass of pendulum 2 in kg


def derivative_solver(pendulem_conditions, t):

    dydx = np.zeros_like(pendulem_conditions)
    dydx[0] = pendulem_conditions[1]

    del_ = pendulem_conditions[2] - pendulem_conditions[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*pendulem_conditions[1]*pendulem_conditions[1]*sin(del_)*cos(del_) +
               M2*G*sin(pendulem_conditions[2])*cos(del_) +
               M2*L2*pendulem_conditions[3]*pendulem_conditions[3]*sin(del_) -
               (M1 + M2)*G*sin(pendulem_conditions[0]))/den1

    dydx[2] = pendulem_conditions[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*pendulem_conditions[3]*pendulem_conditions[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(pendulem_conditions[0])*cos(del_) -
               (M1 + M2)*L1*pendulem_conditions[1]*pendulem_conditions[1]*sin(del_) -
               (M1 + M2)*G*sin(pendulem_conditions[2]))/den2

    return dydx


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.02
t = np.arange(0.0, 100, dt)

# ia1 and ia2 are the initial angles (degrees)
# iv1 and iv2 are the initial angular velocities (degrees per second)
ia1 = 90.0
iv1 = 1.0
ia2= -10.0
iv2 = 0.0

# initial pendulem_conditions
pendulem_conditions = np.radians([ia1, iv1, ia2, iv2])

# integrate the ODE using scipy.integrate.
y = integrate.odeint(derivative_solver, pendulem_conditions, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform = ax.transAxes)




ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()
