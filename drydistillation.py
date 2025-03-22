import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('clarke-default')
from scipy.optimize import curve_fit

# functional form and Zr data from Caltech MMRC website
# https://mmrc.caltech.edu/PVD/manuals/Metals%20Vapor%20pressure.pdf
def p_vapor(T,A,B,C,D):
    return A + B/T + C*np.log10(T) + D/T**3

params_Zr = (10.08,-31512,-0.789,0) # MMRC

p_Ra_ref = np.logspace(0,5,6)/101325.
T_Ra_ref = np.array((819,906,1037,1209,1446,1799)) # wikipedia
T_Zr_ref = np.array((2639,2891,3197,3575,4053,4678)) # wikipedia

temps = np.linspace(500,2200,1000)

popt_Ra,pcov_Ra = curve_fit(p_vapor,T_Ra_ref,np.log10(p_Ra_ref))
p_Ra = 10**(p_vapor(temps, *popt_Ra))

p_Zr = 10**(p_vapor(temps, *params_Zr))

# fit to wikipedia data for Zr as a comparison
popt_Zr,pcov_Zr = curve_fit(p_vapor,T_Zr_ref,np.log10(p_Ra_ref),p0 = params_Zr)
p_Zr_2 = 10**(p_vapor(temps, *popt_Zr))

fig,ax = plt.subplots()
ax.semilogy(temps,p_Zr,label='Zr vapor')
#plt.semilogy(temps,p_Zr_2,label='Zr2') check on consistency between data from wikipedia and Caltech MMRC
ax.semilogy(temps,p_Ra,label='Ra vapor')
ax.axvline(2128,ls='--',label='Zr melting point',color = next(ax._get_lines.prop_cycler)['color'])
ax.axvline(973,ls='--',label='Ra melting point',color = next(ax._get_lines.prop_cycler)['color'])
ax.axvline(2010,ls='--',label='Ra boiling point',color = next(ax._get_lines.prop_cycler)['color'])
ax.set_title('Vapor pressure for solid Ra + Zr')
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('Vapor pressure [atm]')
ax.set_xlim([500,2200])
ax.set_ylim([1e-20,1e1])
ax.legend(loc='best')
plt.show()