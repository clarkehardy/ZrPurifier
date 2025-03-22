import numpy as np

T_half_U238 = 1.41e17 # seconds
M_U238 = 238.051 # g/mol
T_half_Th232 = 4.434e17 # seconds
M_Th232 = 232.038 # g/mol
N_avogadro = 6.02e23 # molecules

def ppb_U238(act_Ra226):
    # takes activity of Ra226 in Bq
    # returns concentration in ppb (1e9 p/b x 1e-3 g/kg)
    return act_Ra226*T_half_U238*M_U238*1e6/np.log(2)/N_avogadro

def ppb_Th232(act_Ra228):
    # takes activity of Ra226 in Bq
    # returns concentration in ppb (1e9 p/b x 1e-3 g/kg)
    return act_Ra228*T_half_Th232*M_Th232*1e6/np.log(2)/N_avogadro

def activity_Ra226(conc_U238):
    # takes concentration of U238 in ppb
    # returns activity of Ra226 in Bq
    return N_avogadro*np.log(2)/T_half_U238/M_U238*conc_U238*1e-6

print(ppb_U238(0.318e-3))
print(ppb_U238(0.109e-3))
print(ppb_U238(0.373e-3))
print(ppb_U238(0.089e-3))