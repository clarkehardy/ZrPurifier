import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sp
import scipy.stats as st

#****************************************************************************
# SET COLUMN DESIGN PARAMETERS HERE
#****************************************************************************
dpellet = 2.0e-3 # particle diameter, m
# dpellet is equivalent to dps used in Moghaddam et al.
phi = 0.65 # actual packing fraction
rho_Zr = 6.49 # density of solid Zr, g/cm^3
m_Zr = 1. # mass of Zr, kg
vol = m_Zr*1e-3/(rho_Zr*phi) # volume of the column, m^3
dtube = 1.375*0.0254 # diameter of the column in m
Ltube = 4*vol/np.pi/dtube**2 # length of the column, m
mdot = 1.37*5.887/60. # recirculation mass flow rate, g/s
Mtotal = 1.5e3 # total xenon mass, g
print('Total length of Zr pellet column: {:.2f} inches'.format(Ltube*100./2.54))

#****************************************************************************
# RANDOM WALK COMPUTATION OF COLLISIONS
#****************************************************************************
dXe = 396e-12 #432e-12 # 432 picometers
dO2 = 346e-12 #304e-12 # 304 picometers
MXe = 131.293 # g/mol
MO2 = 32.0 # g/mol
kB = 1.381e-23 # J/K
T = 293. # K
p = 101.3e3 # Pa
l = 4*kB*T/(np.sqrt(1+MO2/MXe)*np.pi*(dXe+dO2)**2*p) # mean free path in meters
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
rho = 5.887e3 # density of xenon in g/m^3
vflow = mdot/(rho*np.pi*dtube**2/4.) # flow speed through column in m/s
# vflow is also called superficial velocity and is equal to Q/A
ttransit = Ltube/vflow # transit time through column
Ncollisions = ttransit/tcollision # number of collisions with a pellet
print('Mean free path of an O2 molecule: {:.2f} nm'.format(l*1e9))
print('Average speed of an O2 molecule: {:.1f} m/s'.format(vav))
print('Number of steps to reach nearest pellet: {:.3e}'.format(Nsteps))
print('Average time between molecule-molecule collisions: {:.0f} ps'.format(tau*1e12))
print('Average time between molecule-pellet collisions: {:.0f} us'.format(tcollision*1e6))
print('Average transit time for a molecule: {:.2f} s'.format(ttransit))
print('Average number of collisions for a transiting molecule: {:.2f}'.format(Ncollisions))

# multiply number of collisions with sticking efficiency and
# fraction of area unscathed to get probability of sticking
stick_eff = 0.10 # sticking efficiency
Vpore = 4.*np.pi*xexp**3./3. # volume in m^3
void_total = vol*phi # total void volume in m^3
Npores = void_total/Vpore # total number of pores
Vpellet = dpellet*np.pi*dpellet**2/4.
Npellets = vol*phi/Vpellet
poreperpel = Npores/Npellets
print('Average number of pores per pellet: {:.0f}'.format(poreperpel))

conc = 30e-9 # 1 ppb to start
NAv = 6.022e23 # Avogadro's number
Nimpur = conc*Mtotal*NAv/MXe # number of impurities in the xenon
Qimpur = conc*mdot*NAv/MXe # impurities flowing through per second
cover = 1e13*1e6 # molecules per mm^2 to m^2, value from meeting with Paolo
Nlayers = 10. # number of monolayers of oxygen in Zr surface
unit_SA = 1.5*np.pi*dpellet**2 # area of a single pellet, m^2
total_SA = unit_SA*Npellets # total surface area of all pellets, m^2
places = cover*total_SA*Nlayers # available places for an impurity to stick
tcirc = Mtotal/mdot # time to recirculate all xenon
# dosing the xenon with air
R = 8.314 # gas constant in J/mol/K
pa_per_torr = 133.32 # convert Pascals to torr
p_dose = 1.*pa_per_torr # pressure of dosing volume converted to Pa
O2_fraction = 0.21 # fraction of O2 in air
V_dose = cover*total_SA*R*T*1000./p_dose/NAv/O2_fraction # dosing volume in mL
print('Time to recirculate all xenon: {:.2f} hours'.format(tcirc/3600.))
print('Number of places for an impurity to stick: {:.2e}'.format(places))
print('Equivalent air volume to saturate purifier: {} mL'.format(V_dose))
print('Equivalent concentration to saturate purifier: {} ppb'.format())

#****************************************************************************
# COMPUTE RELEVANT FLOW PARAMETERS
#****************************************************************************
mu = 2.28e-5 # xenon viscosity, Pa.s
A = 190. # constant A in Ergun equation
Nratio = dtube/dpellet # tube to pellet diameter ratio
M = 1.+2./(3.*Nratio*(1.-eps)) # correction to Re for wall effects
B = (2./Nratio**2+0.77)**2 # constant B in Ergun equation
pa_to_torr = 7.50062e-3 # unit conversion factor

# function describing pressure drop
def Pdrop(U):
    return pa_to_torr*psi(U)*Ltube*((1.-eps)/eps**3)*rho*U**2*M/dpellet

# dimensionless friction factor
def psi(U):
    return A/Re(U)+B

# compute Reynolds number
def Re(U):
    return (rho*1e-3)*U*dpellet/(mu*(1-eps)*M)

# compute permeability
def perm(U):
    return U*mu*Ltube*pa_to_torr/Pdrop(U)

print('Reynolds number for this flow: {:.1f}'.format(Re(vflow)))
print('Pressure drop over column: {:.1f} torr'.format(Pdrop(vflow)))
print('Permeability of the Zr: {} m^2'.format(perm(vflow)))

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
    conc_impur.append(Nimpur*MXe/Mtotal/NAv)
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

plt.show()
