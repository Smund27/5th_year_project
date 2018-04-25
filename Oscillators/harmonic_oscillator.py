import numpy as np 
import matplotlib.pyplot as plt

class harmonic_oscillator:

	def __init__(self, dt, x0, xd0, omega_squared):

		self.dt = dt
		self.dt_squared = dt**2
		self.x0 = x0
		self.t = dt
		self.omega_squared = omega_squared
		self.xd0 = xd0
		self.x = [x0 + xd0 * dt, x0]

	def step(self):

		xn, xn_minus1 = self.x
		xn_plus1 = (2 - self.omega_squared * self.dt_squared) * xn -  xn_minus1
		self.x = (xn_plus1, xn)
		self.t += self.dt

	def number_steps(self, tmax, niter):
	
		ts = [self.t]
		vals = [self.x[0]]
		while self.t < tmax:
			for i in range(niter):
				self.step()
			vals.append(self.x[0])
			ts.append(self.t)

		return np.array(ts), np.array(vals)

x = harmonic_oscillator(1e-5, 1, 0, 1)

ts, vals = x.number_steps(20, 5)
plt.plot(ts, vals)
plt.show()
	
