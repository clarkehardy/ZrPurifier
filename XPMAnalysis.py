import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import matplotlib.dates as mdates
import csv
import pandas as pd
import datetime
import sys
sys.path.insert(0,'../Scripts/')
import ElectronAttachment
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy import interpolate
from scipy import stats as st
from scipy.integrate import cumtrapz

class XPMDataset():

    def __init__(self,path='XPM_data/scp_postdose2.txt',saes=False,Vdrift=250.,Pxenon=1900):
        print('\nLoading dataset from '+path+'...')
        self.Vdrift = Vdrift # drift field in V/cm (shouldn't ever change)
        self.Pxenon = Pxenon # average Xe pressure in torr; used to convert EL -> concentration
        headers = ['Tc','Ta','Tcrise','Tarise','Cathode','Anode','Offset',\
                   'LVTimestamp','UV','IR','chi2','Nwaveforms','Ntriggers']
        data = []
        with open(path,'r') as infile:
            reader = csv.reader(infile)
            for line in reader:
                data.append(line)
        self.data = pd.DataFrame(data,columns=headers,dtype=float)
        self.data['EL'] = self.EL(self.data['Ta'],self.data['Tc'],self.data['Anode'],self.data['Cathode'])
        self.data['Time'] = [datetime.datetime.fromtimestamp(time-2335132800)\
                             for time in self.data['LVTimestamp'].values]
        # timestamp offset found through trial and error; NOT 1/1/1900 or 1904 as suggested
        self.data['Timestamp'] = [datetime.datetime.timestamp(time) for time in self.data['Time']]
        print('\nDone.\n')
        print(self.data.head())

    def reduce_data(self,num_mc=100,Q_err=0.1):
        # create dataframe of reduced quantities with errors calculated using MC method
        # num_mc: number of normally-distributed points around each Qc or Qa measurement
        # used to compute distribution of EL values
        # Q_err: erro on cathode and anode signals in mV

        print('\nReducing dataset...')
        
        reduced_headers = ['EL','ELErrorLower','ELErrorUpper','Concentration','ConcentrationErrorLower',\
                           'ConcentrationErrorUpper','Cathode','Anode','Timestamp','Time']
        reduced = []
        # find groups of 10 closely-spaced events; if continuous data taking, group into
        # tens starting from the first event
        next_index = 10
        for i in range(len(self.data.index)):
            if i==next_index:
                for j in range(10):
                    if self.data['Timestamp'][j+i-9]-self.data['Timestamp'][j+i-10] < 300:
                        next_index = i+j+1
                    else:
                        next_index += 1
                        break
                evt_group = np.arange(i-10,next_index-10)
                this_group = []
                for j in evt_group:
                    this_group.append(self.get_EL_mc(j,num_mc=num_mc,Q_err=Q_err))
                this_group = self.combine_mc_dists(this_group)
                modeEL, minus1sEL, plus1sEL = self.get_EL_uncertainties(this_group)
                modeConc, minus1sConc, plus1sConc = self.get_conc_uncertainties(this_group)
                this_timestamp = np.mean(self.data['Timestamp'][i-10:next_index-10])
                this_time = datetime.datetime.fromtimestamp(this_timestamp)
                cathode = np.mean(self.data['Cathode'][i-10:next_index-10])
                anode = np.mean(self.data['Anode'][i-10:next_index-10])
                reduced.append([modeEL,minus1sEL,plus1sEL,modeConc,minus1sConc,plus1sConc,\
                                cathode,anode,this_timestamp,this_time])
        self.reduced = pd.DataFrame(reduced,columns=reduced_headers,dtype=float)
        print('\nDone.\n')
        print(self.reduced.head())
        
    def get_array(self,quantity,start,end):
        # takes a string specifying which column of the reduced dataframe to look at and
        # datetime objects specifying the start and end times of the interval
        # returns an array of the desired quantity
        return self.reduced[quantity][(self.reduced['Timestamp']>start) & (self.reduced['Timestamp']<end)].values

    def EL(self,ta,tc,Qa,Qc):
        # Return electron lifetime for given cathode and anode voltages
        # if either the cathode or anode amplitude is negative, return 0
        el = (ta-tc)/np.log(Qc/Qa)
        if isinstance(el,np.float64):
            if Qc<0 or Qa<0:
                return 0
            else:
                return el
        el[(Qc<0) | (Qa<0)] = 0
        return el

    def get_EL_mc(self,index,num_mc=100,Q_err=0.1):
        # Create MC distribution of EL values using uncertainties
        # on anode and cathode voltages

        ta = self.data['Ta'][0] # times should all be the same, so just take the first entry
        tc = self.data['Tc'][0]

        Qc_meas = self.data['Cathode'][index]
        Qa_meas = self.data['Anode'][index]
        Qc_mc = np.random.normal(Qc_meas,Q_err,num_mc)
        Qa_mc = np.random.normal(Qa_meas,Q_err,num_mc)
        EL_meas = []
        for Qa_val in Qa_mc:
            for Qc_val in Qc_mc:
                EL_meas.append(self.EL(ta,tc,Qa_val,Qc_val))
        EL_meas = np.array(EL_meas)
        EL_meas = np.delete(EL_meas,EL_meas<0)
        EL_meas = np.delete(EL_meas,EL_meas>100000)
        return Qa_meas,Qc_meas,EL_meas

    def combine_mc_dists(self,dists):
        # For a single set of acquisitions, combine and histogram
        # the EL distributions
        EL_meas = []
        Qa_meas = []
        Qc_meas = []
        for i in range(len(dists)):
            EL_meas.append(dists[i][2])
            Qa_meas.append(dists[i][0])
            Qc_meas.append(dists[i][1])
        return np.array(Qa_meas),np.array(Qc_meas),np.concatenate(EL_meas)

    def get_EL_uncertainties(self,dist):
        # Get median and uncertainties for a distribution of EL values
        Qa_meas, Qc_meas, EL_meas = dist
        lower_bin = np.mean(EL_meas) - 3*np.std(EL_meas)
        upper_bin = np.mean(EL_meas) + 3*np.std(EL_meas)
        if lower_bin<0: # or lower_bin>100000:
            lower_bin = 0
        if upper_bin>100000: #<0:
            upper_bin = 100000
        bin_array = np.linspace(lower_bin,upper_bin,100)
        cts,edges = np.histogram(EL_meas,bins=bin_array,density=True)
        bins = (edges[:-1]+edges[1:])/2.
        lower = bins[np.argmin(abs(cumtrapz(cts,bins)-0.16))]
        upper = bins[np.argmin(abs(cumtrapz(cts,bins)-0.84))]
        mode = bins[np.argmin(abs(cumtrapz(cts,bins)-0.5))]
        #mode = bins[np.argmax(cts)]
        return mode, mode-lower, upper-mode

    def get_conc_uncertainties(self,dist):
        # Get median and uncertainties for a distribution of concentration values
        Qa_meas, Qc_meas, EL_meas = dist
        conc_meas = ElectronAttachment.Concentration(EL_meas,self.Vdrift,self.Pxenon)
        conc_meas[EL_meas==0] = 1e9
        lower_bin = np.mean(conc_meas) - 3*np.std(conc_meas)
        upper_bin = np.mean(conc_meas) + 3*np.std(conc_meas)
        if lower_bin<0: # or lower_bin>100000:
            lower_bin = 0
        #if upper_bin>100000: #<0:
        #    upper_bin = 100000
        bin_array = np.linspace(lower_bin,upper_bin,100)
        cts,edges = np.histogram(conc_meas,bins=bin_array,density=True)
        bins = (edges[:-1]+edges[1:])/2.
        lower = bins[np.argmin(abs(cumtrapz(cts,bins)-0.16))]
        upper = bins[np.argmin(abs(cumtrapz(cts,bins)-0.84))]
        mode = bins[np.argmin(abs(cumtrapz(cts,bins)-0.5))]
        #mode = bins[np.argmax(cts)]
        return mode, mode-lower, upper-mode

# standalone variables and functions

rhoXeSTP = 5.894 # xenon density at STP, g/L
mXe = 131.29 # xenon molar mass, g/mol
mLXe = 1500. # total liquid xenon mass, g

def ConcentrationModel(t,x0,ef,Lambda):
    return x0*np.exp(-ef*t/tauC)+Lambda*tauC/ef

def chi2(data,model,error):
        # Return chi2 for a fit
        return np.sum((data - model)**2/error**2)

def chi2_asymm(data,model,errorLower,errorUpper):
        # Return chi2 for a fit to data with asymmetric errors
        errorLower[model>data] = errorUpper[model>data]
        return np.sum((data - model)**2/errorLower**2)

def driver_function(params,xobs,yobs,yerrlower,yerrupper,fitfunc):
        # Returns the chi2 to be minimized for an arbitrary fitting function
        ynew = fitfunc(xobs,*params)
        return chi2_asymm(yobs,ynew,yerrlower,yerrupper)

def efficiency(ef,deltaef,f,deltaf):
    e = ef/f
    deltae = np.sqrt((deltaef/f)**2+(ef*deltaf/f**2)**2)
    print('\nEfficiency: {:.3f}+/-{:.3f}'.format(e,deltae))
    return e,deltae

if __name__=='__main__':

    SCP = XPMDataset('XPM_data/scp_postdose2.txt')
    SCP.reduce_data(Q_err=0.01)
    
    SAES = XPMDataset('XPM_data/saes_postdose1.txt')
    SAES.reduce_data(Q_err=0.01)

    # fit data for the SCP
    VDot = 1.34 # xenon flow rate, SLPM
    mDot = VDot*rhoXeSTP/60. # xenon mass flow rate, g/s
    tauC = mLXe/mDot # total mass over mass flow rate, ie LXe turnover time [s]
    
    scp_start = datetime.datetime.timestamp(datetime.datetime(2023,3,7,10))
    scp_end = datetime.datetime.timestamp(datetime.datetime(2023,3,9,10))
    scp_times = SCP.get_array('Timestamp',scp_start,scp_end)
    scp_times = scp_times - scp_times[0] # want the time since the start of the interval, not the UNIX timestamp

    scp_concs = SCP.get_array('Concentration',scp_start,scp_end)
    scp_conc_lower = SCP.get_array('ConcentrationErrorLower',scp_start,scp_end)
    scp_conc_upper = SCP.get_array('ConcentrationErrorUpper',scp_start,scp_end)
    scp_conc_errors = (scp_conc_lower + scp_conc_upper)/2.

    p0 = [8.,1.0,1.]
    p,c = curve_fit(ConcentrationModel,scp_times,scp_concs,sigma=scp_conc_errors,absolute_sigma=True,p0=p0)
    e = [np.sqrt(c[i][i]) for i in range(c.shape[0])]

    ef = p[1]
    deltaef = e[1]
    
    print('\nSCP:')
    print('Initial O2 concentration: {:.2f}+/-{:.2f} ppb'.format(p[0],e[0]))
    print('Equilibration factor times efficiency: {:.2f}+/-{:.2f}'.format(p[1],e[1]))
    print('Constant O2 addition rate: {:.3f}+/-{:.3f} ppt/hour'.format(p[2]*3600.*1e3,3600.*1e3*e[2]))

    chi2Conc = chi2(scp_concs,ConcentrationModel(scp_times,*p),scp_conc_errors)
    print('chi2/ndof: {:.3f}/{:.0f}'.format(chi2Conc,len(scp_concs)-len(p)-1))
    print('prob: {:.4f}'.format(1-st.chi2.cdf(chi2Conc,len(scp_concs)-len(p)-1)))

    # plot the result for the SCP
    timeArray = np.linspace(scp_times[0],scp_times[-1],1000)
    concArray = ConcentrationModel(timeArray,*p)

    fig,ax = plt.subplots()
    ax.errorbar(scp_times/3600.,scp_concs,yerr=[scp_conc_lower,scp_conc_upper],fmt='.',label='Purification raw')
    ax.plot(timeArray/3600.,concArray,label='Purification fit')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Equivalent O$_2$ Concentration [ppb]')
    textstr = '\n'.join((
        r'$x_0={:.2f}\pm{:.2f}$ ppb'.format(p[0],e[0]),
        r'$\varepsilon f={:.2f}\pm{:.2f}$'.format(p[1],e[1]),
        r'$\Lambda/n={:.2f}\pm{:.2f}$ ppt/h'.format(p[2]*3600.*1e3,3600.*1e3*e[2]),
        r'$\chi^2/$ndof$={:.2f}/{:.0f}$'.format(chi2Conc,len(scp_concs)-len(p)-1),
        r'prob$={:.4f}$'.format(1-st.chi2.cdf(chi2Conc,len(scp_concs)-len(p)-1))))
    props = dict(boxstyle='square', alpha=0.2)
    ax.text(0.6*ax.get_xlim()[-1],0.7*ax.get_ylim()[-1],textstr,bbox=props)
    fig.savefig('xpm_purification_scp.png',bbox_inches='tight')
    fig.tight_layout()

    # fit data for the SAES
    VDot = 1.21 # xenon flow rate, SLPM
    mDot = VDot*rhoXeSTP/60. # xenon mass flow rate, g/s
    tauC = mLXe/mDot # total mass over mass flow rate, ie LXe turnover time [s]
    
    saes_start = datetime.datetime.timestamp(datetime.datetime(2023,3,4,0))
    saes_end = datetime.datetime.timestamp(datetime.datetime(2023,3,6,11))
    saes_times = SAES.get_array('Timestamp',saes_start,saes_end)
    saes_times = saes_times - saes_times[0] # want the time since the start of the interval, not the UNIX timestamp

    saes_concs = SAES.get_array('Concentration',saes_start,saes_end)
    saes_conc_lower = SAES.get_array('ConcentrationErrorLower',saes_start,saes_end)
    saes_conc_upper = SAES.get_array('ConcentrationErrorUpper',saes_start,saes_end)
    saes_conc_errors = (saes_conc_lower + saes_conc_upper)/2.

    p0 = [8.,1.0,1.]
    p,c = curve_fit(ConcentrationModel,saes_times,saes_concs,sigma=saes_conc_errors,absolute_sigma=True,p0=p0)
    e = [np.sqrt(c[i][i]) for i in range(c.shape[0])]

    f_saes = p[1]
    df_saes = e[1]
    
    print('\nSAES:')
    print('Initial O2 concentration: {:.2f}+/-{:.2f} ppb'.format(p[0],e[0]))
    print('Equilibration factor times efficiency: {:.2f}+/-{:.2f}'.format(p[1],e[1]))
    print('Constant O2 addition rate: {:.3f}+/-{:.3f} ppt/hour'.format(p[2]*3600.*1e3,3600.*1e3*e[2]))

    chi2Conc = chi2(saes_concs,ConcentrationModel(saes_times,*p),saes_conc_errors)
    print('chi2/ndof: {:.3f}/{:.0f}'.format(chi2Conc,len(saes_concs)-len(p)-1))
    print('prob: {:.4f}'.format(1-st.chi2.cdf(chi2Conc,len(saes_concs)-len(p)-1)))

    # plot the result for the SAES
    timeArray = np.linspace(saes_times[0],saes_times[-1],1000)
    concArray = ConcentrationModel(timeArray,*p)

    fig,ax = plt.subplots()
    ax.errorbar(saes_times/3600.,saes_concs,yerr=[saes_conc_lower,saes_conc_upper],fmt='.',label='Purification raw')
    ax.plot(timeArray/3600.,concArray,label='Purification fit')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Equivalent O$_2$ Concentration [ppb]')
    textstr = '\n'.join((
        r'$x_0={:.2f}\pm{:.2f}$ ppb'.format(p[0],e[0]),
        r'$\varepsilon f={:.2f}\pm{:.2f}$'.format(p[1],e[1]),
        r'$\Lambda/n={:.2f}\pm{:.2f}$ ppt/h'.format(p[2]*3600.*1e3,3600.*1e3*e[2]),
        r'$\chi^2/$ndof$={:.2f}/{:.0f}$'.format(chi2Conc,len(saes_concs)-len(p)-1),
        r'prob$={:.4f}$'.format(1-st.chi2.cdf(chi2Conc,len(saes_concs)-len(p)-1))))
    props = dict(boxstyle='square', alpha=0.2)
    ax.text(0.6*ax.get_xlim()[-1],0.7*ax.get_ylim()[-1],textstr,bbox=props)
    fig.savefig('xpm_purification_saes.png',bbox_inches='tight')
    fig.tight_layout()

    e_saes,de_saes = efficiency(ef,deltaef,f_saes,df_saes)

    fig,ax = plt.subplots()
    ax.axvline([1.],ls='--',lw=2,color='navy',label='SAES')
    ax.errorbar([e_saes],[1],xerr=[de_saes],markersize=10,marker='.',color='plum',label='SCP')
    ax.set_ylim([0,2])
    ax.set_xlim([0,2])
    ax.set_yticks([])
    ax.set_xlabel(r'Efficiency, $\varepsilon$')
    ax.set_title(r'Measurement of $\varepsilon$ for the SCP')
    ax.legend(loc='upper right')
    fig.savefig('eps_plot.png',bbox_inches='tight')
