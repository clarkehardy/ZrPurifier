import numpy as np
from matplotlib import pyplot as plt

eta = 2.28e-5 # Pa.s
dp = 0.002 # m
rho = 5.887 # kg/m^3
mfr = 30.e-3 # kg/min
dt = 1.5*2.54e-2 # m
Lt = 11.*2.54e-2 # m
eps = 0.38
A = 190.
v_inlet = mfr*4/(rho*np.pi*dt**2*60) # m/s
N = dt/dp
M = 1.+2./(3.*N*(1.-eps))
B = (2./N**2+0.77)**2
pa_to_torr = 7.50062e-3

def Pdrop(U):
    return pa_to_torr*psi(U)*Lt*((1.-eps)/eps**3)*rho*U**2*M/dp

def psi(U):
    return A/Re(U)+B

def Re(U):
    return rho*U*dp/(eta*(1-eps)*M)

print('Pressure drop over column: {:.1f} torr.'.format(Pdrop(v_inlet)))
