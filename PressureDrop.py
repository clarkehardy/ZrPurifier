import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
style.use('clarke-default')
import scipy.special as sp
import scipy.stats as st
import sys
sys.path.insert(0,'../../Scripts/')
from ElectronAttachment.ElectronAttachment import GXeDensity

# unit conversions and xenon properties
g_per_kg = 1e3 # grams per kilogram
L_per_m3 = 1e3 # litres per meter^3
Pa_per_torr = 133.3 # Pa per torr
Pa_per_bar = 1e5 # Pascals per bar
M_Xe = 131.29 # mass of xenon in grams per mole
rho_Xe_stp = 5.894 # density of Xe gas at STP, g/L

# pellet properties
# dpellet = 4e-3 # diameter of an ST707 pellet in meters
# hpellet = 2e-3 # height of an ST707 pellet in meters
dpellet = 2e-3 # diameter of a Zr pellet in meters
hpellet = 2e-3 # height of a Zr pellet in meters
SApellet = np.pi*dpellet**2/2.+np.pi*dpellet*hpellet # surface area of the pellet in m^2
dps = np.sqrt(SApellet/np.pi) # diameter of a sphere of equal SA
Vpellet = np.pi*dpellet**2*hpellet/4 # volume of an ST707 pellet in m^3
# mpellet = 130e-6 # mass of an ST707 pellet in kg
rho_zr = 6511. # density of Zr in kg/m^3
mpellet = Vpellet*rho_zr

# friction factor parameters
mu = 2.27e-5 # xenon viscosity, Pa.s
A = 190. # constant A in Ergun equation

class Purifier():

    # all values that you may want to change are arguments
    def __init__(self,dtube,mcolumn,phi,P_Xe,mdot_slpm,dpellet=2e-3,hpellet=2e-3,rhopellet=6511.,mpellet=None):
        SApellet = np.pi*dpellet**2/2.+np.pi*dpellet*hpellet # surface area of the pellet in m^2
        self.dps = np.sqrt(SApellet/np.pi) # diameter of a sphere of equal SA
        self.Vpellet = np.pi*dpellet**2*hpellet/4. # volume of a pellet in m^3
        if mpellet is None:
            self.mpellet = self.Vpellet*rhopellet
        else:
            self.mpellet = mpellet
        self.dtube = dtube
        self.eps = 1. - phi # porosity
        self.Vcolumn = self.Vpellet*mcolumn/self.mpellet/phi # total volume of pellet column in meters^3
        self.Ltube = 4*self.Vcolumn/np.pi/dtube**2 # length of pellet column in meters
        mdot = mdot_slpm*rho_Xe_stp/60. # mass flow rate in g/min
        self.rho_GXe = GXeDensity(P_Xe/Pa_per_torr)*M_Xe*L_per_m3 # density of xenon gas in g/m^3
        self.Uflow = mdot/(self.rho_GXe*np.pi*dtube**2/4.) # free stream velocity in m/s
        N = dtube/self.dps # ratio  of tube diameter to equivalent spherical diameter
        self.M = 1.+2./(3.*N*(1.-self.eps)) # correction to Re for wall effects
        self.B = (2./N**2+0.77)**(-2) # constant B in Ergun equation

    # pressure drop in bar
    def Pdrop(self):
        return self.psi()*self.Ltube*((1.-self.eps)/self.eps**3)*(self.rho_GXe/g_per_kg)*self.Uflow**2*self.M/self.dps/Pa_per_bar

    # dimensionless friction factor
    def psi(self):
        return A/self.Re()+self.B

    # compute Reynolds number
    def Re(self):
        return (self.rho_GXe/g_per_kg)*self.Uflow*self.dps/(mu*(1-self.eps)*self.M)

    # compute permeability
    def perm(self):
        return self.Uflow*mu*self.Ltube/Pa_per_bar/self.Pdrop()

if __name__=="__main__":

    # create purifier object
    pur = Purifier(dtube=0.0254*1.375,mcolumn=0.5,phi=0.64,P_Xe=2.*Pa_per_bar,mdot_slpm=1.5,\
                   dpellet=dpellet,hpellet=hpellet,mpellet=mpellet) # for nEXO: 3 in diameter, 5 kg ST-707, 5 bar, 350 SLPM
    print('\nReynolds number for this flow: {:.1f}'.format(pur.Re()))
    print('Pressure drop over column: {:.4f} bar'.format(pur.Pdrop()))
    print('Permeability of the Zr: {:.2e} m^2\n'.format(pur.perm()))

    # loop through tube diameters and porosities
    dtubes = np.linspace(0.025,0.2,100)
    phis = np.array((0.75,0.65,0.55))
    mcolumn = 0.5
    Pdrops_list = []
    for i in range(len(phis)):
        Pdrops = []
        for j in range(len(dtubes)):
            this_pur = Purifier(dtubes[j],mcolumn,phis[i],2.*Pa_per_bar,1.5,\
                                dpellet=dpellet,hpellet=hpellet,mpellet=mpellet)
            Pdrops.append(this_pur.Pdrop())
            del this_pur
        Pdrops_list.append(Pdrops)

    Vtotal_min = Vpellet*mcolumn/mpellet/max(phis)
    Vtotal_max = Vpellet*mcolumn/mpellet/min(phis)
    LD3 = 1e3*(4*Vtotal_max/3/np.pi)**(1./3)
    LD5 = 1e3*(4*Vtotal_min/5/np.pi)**(1./3)

    fig,ax = plt.subplots()
    ax.semilogy(dtubes*1e3,Pdrops_list[0],lw=1,color='dodgerblue',label='porosity={:.2f}'.format(1-phis[0]))
    ax.semilogy(dtubes*1e3,Pdrops_list[1],lw=2,color='blue',label='porosity={:.2f}'.format(1-phis[1]))
    ax.semilogy(dtubes*1e3,Pdrops_list[2],lw=1,color='navy',label='porosity={:.2f}'.format(1-phis[2]))
    ax.set_title('Pressure drop over 5 kg ST707 at 350 SLPM and 2 bar')
    ax.set_ylabel('Pressure drop [bar]')
    ax.set_xlabel('Column diameter [mm]')
    ax.axvline(LD3,ls='--',color='darkorange',label='L/D=3')
    ax.axvline(LD5,ls='--',color='red',label='L/D=5')
    ax.fill_between([LD5,LD3],1e3,color='violet',alpha=0.2)
    ax.set_xlim([25,200])
    ax.set_ylim([2e-4,2e2])
    ax.legend(loc='best')
    # plt.savefig('Figures/pressure_drop.png')
    plt.show()
