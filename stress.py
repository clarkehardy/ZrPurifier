import numpy as np

# dimensions, in units as measured
CF_od = 1.5 # OD of CF nipple, inches
CF_id = 1.37 # ID of CF nipple, inches
SAES_OD = 38.1 # OD of SAES cartridge, mm
SAES_ID = 34.0 # ID of SAES cartridge, mm
bolt_md = 0.2062 # minor diameter of 1/4-20 bolts on CF flange, in
bolt_nd = 0.25 # nominal bolt diameter, in
bolt_torque = 12. # recommended torque for 2.75in flange, ft-lbs
K = 0.2 # bolt friction factor, dimensionless

# unit conversion factors
mm_per_in = 25.4
bar_per_MPa = 10.
mm_per_m = 1e3
Pa_per_MPa = 1e6
N_per_lb = 4.44822

# data for xenon expansion
a = 4.250 # Van der Waals constant, L^2.bar/mol^2
V_m = 16.115 # molar volume at 1.5 bar and 20 C, from NIST, L/mol

# calculate pressure during expansion using the Van Der Waals equation
def p_final(p_1,T_1,T_2):
    return p_1*T_2/T_1+(a/V_m**2)*(T_2/T_1-1)

# yield strength of 321SS at various temperatures
yield_strength = 241.3 # yield strength at room temperature, MPa
tensile_strength = 620.6 # tensile strength at room temperature, MPa
temp = [24,38,93,149,204,260,316,371,427,482,538,593,649,704,760,816] # temperatures for which strength data is given, degrees C
y_red = [1.00,0.97,0.85,0.76,0.69,0.64,0.61,0.59,0.57,0.57,0.56,0.55,0.53,0.50,0.47,0.40] # yield strength reduction factor at the given temperature
t_red = [1.00,0.97,0.89,0.84,0.83,0.83,0.83,0.83,0.83,0.83,0.79,0.71,0.61,0.49,0.37,0.25] # tensile strength reduction factor at a given temperature

# functions to interpolate strength at an arbitrary temperature
def yield_fun(T):
    return yield_strength*np.interp(T,temp,y_red)

def tensile_fun(T):
    return tensile_strength*np.interp(T,temp,t_red)

# functions to calculate hoop stress
def hoop_thin(p,d_o,d_i):
    return p*d_i/(d_o-d_i)

def hoop_thick(p_i,p_o,d_i,d_o):
    r_i = d_i/2
    r_o = d_o/2
    return (p_i*r_i**2-2*p_o*r_o**2+p_i*r_o**2)/(r_o**2-r_i**2)
    
# calculate hoop stress for CF nipple at 760 C
p_xenon = p_final(1.5,293,273+760)
print('Final pressure after heating to 760 C: {:.3f} bar'.format(p_xenon))
a = 0
print('Final pressure assuming ideal gas law: {:.3f} bar'.format(p_final(1.5,293,273+760)))
print('Hoop stress on CF nipple: {:.3f} MPa'.format(hoop_thin(p_xenon/bar_per_MPa,CF_od,CF_id)))

# calculate stress on CF flange bolts from torque applied
force_clamp = bolt_torque*12*N_per_lb/(K*bolt_nd) # lbf
stress_clamp = force_clamp/(np.pi*(bolt_md*mm_per_in/mm_per_m)**2/4)/Pa_per_MPa # MPa
print('Clamping force on each CF bolt from torquing: {:.3f} N'.format(force_clamp))
print('Stress on each bolt from clamping force: {:.3f} MPa'.format(stress_clamp))

# calculate stress on CF flange bolts at 760 C
force_flange = p_xenon/bar_per_MPa*Pa_per_MPa*np.pi*(CF_id*mm_per_in/mm_per_m)**2/4 # N
stress_flange = force_flange/(6*np.pi*(bolt_md*mm_per_in/mm_per_m)**2/4)/Pa_per_MPa # MPa
print('Force applied to end flange at 760 C: {:.3f} N'.format(force_flange))
print('Stress on each of the CF bolts: {:.3f} MPa'.format(stress_flange))

# calculate yield and tensile stress at 760 C for the CF nipple
print('Yield stress at 760 C: {:.3f} MPa'.format(yield_fun(760)))
print('Tensile stress at 760 C: {:.3f} MPa'.format(tensile_fun(760)))
print('Yield stress at 550 C: {:.3f} MPa'.format(yield_fun(550)))
print('Tensile stress at 550 C: {:.3f} MPa'.format(tensile_fun(550)))

# calculate torque at 98% yield stress
yield_304 = 289.6 # MPa
force_max = yield_304*Pa_per_MPa*(np.pi*(bolt_nd*mm_per_in/mm_per_m)**2/4)/N_per_lb # lbs
torque_max = K*(bolt_nd/12)*0.98*force_max # ft-lbs
print('Max torque allowed: {:.3f} ft-lbs'.format(torque_max))
