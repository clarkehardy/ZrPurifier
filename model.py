import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sp
import scipy.stats as st
import sys
sys.path.insert(0,'../Scripts/')
import ElectronAttachment

#****************************************************************************
# SET XPM MEASURED QUANTITIES HERE
#****************************************************************************

# unit conversion factors
mL_per_L = 1e3 # mL per L
L_per_m3 = 1e3 # L per m^3
mL_per_m3 = mL_per_L*L_per_m3 # mL per m^3
Pa_per_torr = 133.3 # Pa per torr
g_per_kg = 1000. # g per kg

# measured values
V_gas = 13.9 # L, taken from rough tubing measurements 10/31/2022
m_Xe = 1.5 # kg, mass of xenon in the XPM from Peter
T = 293. # K, ambient temperature
#P = 1.013e5 # Pa, xenon pressure
P_Xe = 1900.*Pa_per_torr # Pa, xenon pressure
V_dose = 20. # mL, volume of dosing section
E_xpm = 250. # V/cm, E-field in XPM

# properties and fundamental constants
rho_Xe = 3.1 # g/mL, density of LXe
V_liquid = m_Xe/rho_Xe # L, calculated volume of LXe
K_H = 62.5 # dimensionless, Henry's law constant for O2 in xenon, calculated in a paper referenced in Joey Howlett's thesis
M_Xe = 131.29 # g/mol, xenon molar mass
V_MXe = M_Xe/rho_Xe/mL_per_L # L/mol, xenon molar volume
R = 8.314 # J/mol/K, gas constant
O2_fraction = 1. # dimensionless, fraction of O2 in dosing gas (0.21 if air, 1.0 if pure O2)

def n_O2(x_L):
    # returns number of molecules of O2
    # argument: O2 concentration in liquid, [mol/mol]
    # liquid concentration times (moles Xe in liquid, plus moles Xe in gas times Henry's law constant)
    return x_L*(V_liquid/V_MXe + K_H*P_Xe*V_gas/R/T/L_per_m3)

def conc_O2(num_O2):
    # returns O2 concentration in liquid, [mol/mol]
    # arguments: number of molecules of O2
    return num_O2/n_O2(1.)

def P_air(x_L):
    # returns pressure of air in dosing volume required, mbar
    # argument: O2 concentration in liquid, [mol/mol]
    # use ideal gas law to get moles as air pressure in dosing volume
    return n_O2(x_L) * R*T*mL_per_m3/V_dose/O2_fraction/Pa_per_torr

# as a benchmark, use 1/tau=x_O2/257us.ppb, valid for 140 V/cm drift field
tau = 1. #2.386 #50. # us, desired EL from dosing
x_O2 = ElectronAttachment.Concentration(tau,E_xpm)*1e-9
print('For {:.0f} us EL, set dosing volume to {:.2f} torr.\n'.format(tau,P_air(x_O2)))

#****************************************************************************
# SET COLUMN DESIGN PARAMETERS HERE
#****************************************************************************
dpellet = 2.0e-3 # particle diameter, m
# dpellet is equivalent to dps used in Moghaddam et al.
#phi = 0.65 # actual packing fraction
phi = 0.65
rho_Zr = 6.49 # density of solid Zr, g/cm^3
m_Zr = 1. # mass of Zr, kg
vol = m_Zr*1e-3/(rho_Zr*phi) # volume of the column, m^3
print('******')
print(vol)
print('******')
dtube = 1.375*0.0254 # diameter of the column in m
Ltube = 4*vol/np.pi/dtube**2 # length of the column, m
mdot = 1.32*5.887/60. # recirculation mass flow rate, g/s
Mtotal = 1.5e3 # total xenon mass, g
print('Total length of Zr pellet column: {:.2f} inches.\n'.format(Ltube*100./2.54))

#****************************************************************************
# RANDOM WALK COMPUTATION OF COLLISIONS
#****************************************************************************
dXe = 396e-12 #432e-12 # 432 picometers
dO2 = 346e-12 #304e-12 # 304 picometers
MXe = 131.293 # g/mol
MO2 = 32.0 # g/mol
kB = 1.381e-23 # J/K
l = 4*kB*T/(np.sqrt(1+MO2/MXe)*np.pi*(dXe+dO2)**2*P_Xe) # mean free path in meters
mO2 = MO2*1.66e-27 # mass of O2 molecule in kg
vav = np.sqrt(8*kB*T/(np.pi*mO2)) # average speed of impurity in m/s
eps = 1 - phi # void fraction
xexp = 0.3691*1e-3 # mean distance particle must travel before
# colliding with a pellet; equal to average pore radius from
# Wenli Zhang thesis table 5.2
Nsteps = xexp**2*sp.gamma(3./2.)**2*3./2./l**2 # number of steps for
# a random walk of length xrms
tau = l/vav # molecule-molecule collision time
tcollision = Nsteps*tau # molecule-pellet collision time
#rho = 5.887e3 # density of xenon in g/m^3
rho_GXe = ElectronAttachment.GXeDensity(P_Xe/Pa_per_torr)*M_Xe*L_per_m3 # density of xenon gas in g/m^3
vflow = mdot/(rho_GXe*np.pi*dtube**2/4.) # flow speed through column in m/s
# vflow is also called superficial velocity and is equal to Q/A
ttransit = Ltube/vflow # transit time through column
Ncollisions = ttransit/tcollision # number of collisions with a pellet
print('Mean free path of an O2 molecule: {:.2f} nm'.format(l*1e9))
print('Average speed of an O2 molecule: {:.1f} m/s'.format(vav))
print('Number of steps to reach nearest pellet: {:.3e}'.format(Nsteps))
print('Average time between molecule-molecule collisions: {:.0f} ps'.format(tau*1e12))
print('Average time between molecule-pellet collisions: {:.0f} us'.format(tcollision*1e6))
print('Average transit time for a molecule: {:.2f} s'.format(ttransit))
print('Average number of collisions for a transiting molecule: {:.2f}\n'.format(Ncollisions))

# multiply number of collisions with sticking efficiency and
# fraction of area unscathed to get probability of sticking
stick_eff = 0.10 # sticking efficiency
Vpore = 4.*np.pi*xexp**3./3. # volume in m^3
void_total = vol*(1-phi) # total void volume in m^3
Npores = void_total/Vpore # total number of pores
Vpellet = dpellet*np.pi*dpellet**2/4.
Npellets = vol*phi/Vpellet
poreperpel = Npores/Npellets
print('Average number of pores per pellet: {:.0f}\n'.format(poreperpel))

#conc = x_O2 # 1 ppb to start
NAv = 6.022e23 # Avogadro's number
#Nimpur = conc*Mtotal*NAv/MXe # number of impurities in the xenon
Nimpur = n_O2(x_O2)*NAv
Qimpur = x_O2*mdot*NAv/MXe # impurities flowing through per second
cover = 1e13*1e6 # molecules per mm^2 to m^2, value from meeting with Paolo
Nlayers = 5. # number of monolayers of oxygen in Zr surface
unit_SA = 1.5*np.pi*dpellet**2 # area of a single pellet, m^2
total_SA = unit_SA*Npellets # total surface area of all pellets, m^2
frac_SA = 0.5*total_SA # fraction of the total surface that is pure Zr, m^2
places = cover*frac_SA*Nlayers # available places for an impurity to stick
tcirc = Mtotal/mdot # time to recirculate all xenon
# dosing the xenon with air
R = 8.314 # gas constant in J/mol/K
#p_dose = 760.*Pa_per_torr # pressure of dosing volume converted to Pa
#V_dose = places*R*T*mL_per_m3/p_dose/NAv/O2_fraction # dosing volume in mL
x_sat = conc_O2(places/NAv) # liquid concentration required for full purifier saturation
P_sat = P_air(x_sat) # pressure in dosing volume for full purifier saturation
print('Time to recirculate all xenon: {:.2f} hours'.format(tcirc/3600.))
print('Number of places for an impurity to stick: {:.2e}'.format(places))
print('Equivalent dosing volume pressure to saturate purifier: {:.2f} torr'.format(P_sat))
print('Equivalent concentration to saturate purifier: {:.2f} ppb'.format(x_sat*1e9))
print('Equivalent electron lifetime to saturate purifier: {:.3f} us\n'.format(ElectronAttachment.EL(x_sat*1e9,E_xpm)))

#****************************************************************************
# COMPUTE RELEVANT FLOW PARAMETERS
#****************************************************************************
mu = 2.27e-5 # xenon viscosity, Pa.s
A = 190. # constant A in Ergun equation
Nratio = dtube/dpellet # tube to pellet diameter ratio
M = 1.+2./(3.*Nratio*(1.-eps)) # correction to Re for wall effects
B = (2./Nratio**2+0.77)**(-2) # constant B in Ergun equation

# function describing pressure drop
def Pdrop(U):
    return psi(U)*Ltube*((1.-eps)/eps**3)*(rho_GXe/g_per_kg)*U**2*M/dpellet/Pa_per_torr

# dimensionless friction factor
def psi(U):
    return A/Re(U)+B

# compute Reynolds number
def Re(U):
    return (rho_GXe/g_per_kg)*U*dpellet/(mu*(1-eps)*M)

# compute permeability
def perm(U):
    return U*mu*Ltube/Pa_per_torr/Pdrop(U)

print('Reynolds number for this flow: {:.1f}'.format(Re(vflow)))
print('Pressure drop over column: {:.1f} torr'.format(Pdrop(vflow)))
print('Permeability of the Zr: {:.2e} m^2\n'.format(perm(vflow)))

print(vflow)
print(Ltube)

#****************************************************************************
# NNUMERICALLY CALCULATE SATURATION AND PURIFICATION RATES
#****************************************************************************
clean_frac = 1.
stuck = 0
deltaS = 0
dt = 60. # time increment in s
i = 0
frac_unsat = []
conc_impur = []
sim_time = 72*3600 # number of hours in s
times = np.linspace(0,sim_time,int(round(sim_time/dt)))
Ndiffs = 100 # number of times impurities have been redistributed
# throughout the full volume
# do diffusion continuosly rather than discrete steps

# loop through time steps and recalculate number of impurities
# and saturation fraction
for time in times:
    clean_frac = 1 - stuck/places
    Qimpur = Nimpur*mdot/Mtotal
    frac_unsat.append(clean_frac)
    conc_impur.append(conc_O2(Nimpur/NAv)) #Nimpur*MXe/Mtotal/NAv)
    dS = Qimpur*dt #Qimpur*Ncollisions*clean_frac*stick_eff*dt, almost certain this is wrong
    stuck = stuck + dS
    #deltaS = deltaS + dS
    Nimpur = np.max((Nimpur-dS,0))
    #if np.floor(time/tdiff)>recircs:
    #    Nimpur = np.max((Nimpur-stuck,0))
    #    deltaS = 0
    
frac_unsat = np.array(frac_unsat)
conc_impur = np.array(conc_impur)

plt.figure()
plt.plot(times/3600.,conc_impur*1e9,label='Impurity concentration (ppb)')
plt.ylabel('Impurity Concentration')
plt.xlabel('Time (hours)')
plt.legend(loc='best')
plt.savefig('conc.png',bbox_inches='tight')

plt.figure()
plt.plot(times/3600.,1-frac_unsat,label='Fraction saturated')
plt.ylabel('Satured fraction')
plt.xlabel('Time (hours)')
plt.legend(loc='best')
plt.savefig('sat.png',bbox_inches='tight')

#plt.show()
