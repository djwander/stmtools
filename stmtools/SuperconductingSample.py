from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from NanonisMeasurement import NanonisMeasurement
from Physics import BCS, apply_lock_in_broadening


class NIS:
    """Class containing standard data treatment applied on NIS tunneling spectra"""

    unit_factors = {'T':1e12,'G':1e9,'M':1e6,'k':1e3,'m':1e-3,'u':1e-6,'n':1e-9,'p':1e-12,'a':1e-12,'V':1}

    @staticmethod
    def get_coherence_peaks(spectrum_dIdV):
        """ find coherence peaks using cubic interpolation assuming that they are on opposite sides of x=0
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
        Returns:
            coherence_peaks:                    [ [pt1x,pt1y],[pt2x,pt2y]]
        """
        left_peak = spectrum_dIdV.get_maximum(x_range=(None,0),interpolation="spline",interpolation_args={'k':3})
        right_peak = spectrum_dIdV.get_maximum(x_range=(0,None),interpolation="spline",interpolation_args={'k':3})
        return [left_peak,right_peak]

    @staticmethod
    def get_gap_estimate(spectrum_dIdV):
        """ calculate a rough estimate of the superconducting gap based on the position of the coherence peaks 
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
        Returns:
            Delta (float):                      estimate of the superconducting gap. systematically larger than the true value
        """
        cpks = NIS.get_coherence_peaks(spectrum_dIdV)
        return(cpks[1][0]-cpks[0][0])/2

    @staticmethod
    def get_normal_state_conductance(spectrum_dIdV,n_pts=10):
        """ calculate approximative normal state conductance by averaging the first and last data points
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
            n_pts (int):                        number of data points at both ends to use for averaging
        Returns:
            normal state conductance (float)
        """
        x,y =spectrum_dIdV.get_x_y()
        delta = NIS.get_gap_estimate(spectrum_dIdV)

        mean_y1 = np.mean(y[:n_pts])
        mean_x1 = np.mean(x[:n_pts])
        correction_factor1 = BCS.DOS(mean_x1,delta)

        mean_y2 = np.mean(y[-n_pts:])
        mean_x2 = np.mean(x[-n_pts:])
        correction_factor2 = BCS.DOS(mean_x2,delta)

        return np.mean([mean_y1/correction_factor1,
                        mean_y2/correction_factor2])

    @staticmethod
    def get_peak_increase(spectrum_dIdV,n_pts=10):
        """ calculate the approximate ratio of coherence peak height / normal state conductance
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
            n_pts (int):                        number of data points at both ends to use for averaging
        Returns:
            coherence peak height / normal state conductance (float)
        """
        pks = NIS.get_coherence_peaks(spectrum_dIdV)
        s0 = NIS.get_normal_state_conductance(spectrum_dIdV,n_pts)
        return np.mean([pks[0][1]/s0,pks[1][1]/s0])

    @staticmethod
    def correct_bias_offset(spectrum_dIdV):
        """ shift the bias such that the two coherence peaks are symmetric around 0
        assumes that the shift is smaller than Delta
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
        """
        left_peak,right_peak = NIS.get_coherence_peaks(spectrum_dIdV)
        shift = (right_peak[0]+left_peak[0])/2
        spectrum_dIdV.apply_transformation(lambda x,y:(x-shift,y))
        return spectrum_dIdV

    @staticmethod
    def real_dIdV(x,Delta,T,sig0=1,V_ac_pk=None,V_ac_rms=None,Gamma=0,int_point_factor=10):
        """ dIdV as measured in a real experiment, taking into account temperature effects and smoothing due to the AC Voltage applied by the lock-in
        Arguments:
            Delta (float):                  gap in eV
            T (float):                      temperature in K
            sig0(float):                    normal state conductance. default: 1
            V_ac_pk (float,optional):       peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
            V_ac_rms (float,optional):      root mean squared amplitude of the applied AC voltage in V.
            Gamma(float):                   Dynes parameter in eV
            int_point_factor (integer):     perform the integration on int_point_factor more points than E
        """
        Vac = V_ac_pk if V_ac_pk is not None else V_ac_rms*np.sqrt(2)
        
        E_range = (x[0]-Vac,x[-1]+Vac)
        theory = BCS.dIdV_NIS_eV(E_range,E_res=x[1]-x[0],
                                Delta=Delta,T_N=T,int_point_factor=int_point_factor,Gamma=Gamma)
        if V_ac_pk is not None or V_ac_rms is not None:
            theory = apply_lock_in_broadening(theory,V_ac_pk = V_ac_pk, V_ac_rms=V_ac_rms,integration_factor=int_point_factor)
        return sig0*theory.evaluate(x,interpolation='linear')

    @staticmethod
    def fit_Delta_sigma(spectrum_dIdV,T,sig_guess=None,Delta_guess=None,V_ac_pk=None,V_ac_rms=None,Gamma=0,int_point_factor=10,**kwargs):
        """ Fit the dIdV signal with the curve expected for a BCS superconductor. Free parameters: Delta and sigma0
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
            T (float):                          temperature in K
            sig_guess (float,optional)          initial guess for the normal state conductance. if None: estimated from the data
            Delta_guess (float,optional):       initial guess for the superconducting gap. if None: calculated from the coherence peak positions
            V_ac_pk (float,optional):           peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
            V_ac_rms (float,optional):          root mean squared amplitude of the applied AC voltage in V.
            Gamma(float):                       Dynes parameter in eV
            int_point_factor (integer):         perform the DOS convolution on int_point_factor more points than in the spectrum
            kwargs (dictionary):                additional arguments passed to scipy.curvefit
        Returns:
            (popt,pcov,perr,kwargs) where popt = [sig0,Delta], args is a dictionary holding the fit data, to be passed to NIS.real_dIdV(x,**kwargs)
        """
        spectrum = spectrum_dIdV
        if spectrum_dIdV._x_unit[0] != 'V': # not V but maybe mV - convert it to V
            fac = NIS.unit_factors[spectrum_dIdV._x_unit[0]]
            spectrum = deepcopy(spectrum_dIdV)
            spectrum.apply_transformation(lambda x,y: (x*fac,y))

        p0 = [ NIS.get_normal_state_conductance(spectrum) if sig_guess is None else sig_guess,
               NIS.get_gap_estimate(spectrum) if Delta_guess is None else Delta_guess]
        if not "p0" in kwargs:
            kwargs["p0"] = p0

        if not "epsfcn" in kwargs:
            kwargs['epsfcn'] = 10e-3

        popt,pcov = spectrum.fit(
            lambda x,sig0,Delta:NIS.real_dIdV(x = x,sig0 = sig0,Delta = Delta,T=T,Gamma=Gamma,
                V_ac_rms=V_ac_rms,V_ac_pk=V_ac_pk,int_point_factor=int_point_factor),
                **kwargs)

        return NISFitResult(
            spectrum_dIdV   = spectrum_dIdV,
            Delta           = popt[1],
            T               = T,
            sig0            = popt[0],
            Gamma           = Gamma,
            fit_parameter   = ['sig0','Delta'],
            popt            = popt,
            pcov            = pcov,
            V_ac_pk         = V_ac_pk,
            V_ac_rms        = V_ac_rms
        )

    @staticmethod
    def fit_Delta(spectrum_dIdV,T,sig0 =None,Delta_guess=None,V_ac_pk=None,V_ac_rms=None,Gamma=0,int_point_factor=10,**kwargs):
        """ Fit the dIdV signal with the curve expected for a BCS superconductor. Free parameters: Delta
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
            T (float):                          temperature in K
            sig0 (float,optional)               normal state conductance. if None: estimated from the data
            Delta_guess (float,optional):       initial guess for the superconducting gap. if None: calculated from the coherence peak positions
            V_ac_pk (float,optional):           peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
            V_ac_rms (float,optional):          root mean squared amplitude of the applied AC voltage in V.
            Gamma(float):                       Dynes parameter in eV
            int_point_factor (integer):         perform the DOS convolution on int_point_factor more points than in the spectrum
            kwargs (dictionary):                additional arguments passed to scipy.curvefit
        Returns:
            (popt,pcov,perr,kwargs) where popt = [Delta], args is a dictionary holding the fit data, to be passed to NIS.real_dIdV(x,**kwargs)
        """
        spectrum = spectrum_dIdV
        if spectrum_dIdV._x_unit[0] != 'V': # not V but maybe mV - convert it to V
            fac = NIS.unit_factors[spectrum_dIdV._x_unit[0]]
            spectrum = deepcopy(spectrum_dIdV)
            spectrum.apply_transformation(lambda x,y: (x*fac,y))

        sig0 = sig0 if sig0 is not None else NIS.get_normal_state_conductance(spectrum)

        p0 = [ NIS.get_gap_estimate(spectrum) if Delta_guess is None else Delta_guess]

        if not "p0" in kwargs:
            kwargs["p0"] = p0

        if not "epsfcn" in kwargs:
            kwargs['epsfcn'] = 10e-3

        popt,pcov = spectrum.fit(
            lambda x,Delta:NIS.real_dIdV(x = x,sig0 = sig0,Delta = Delta,T=T,Gamma=Gamma,
                V_ac_rms=V_ac_rms,V_ac_pk=V_ac_pk,int_point_factor=int_point_factor),
                **kwargs)
        
        return NISFitResult(
            spectrum_dIdV   = spectrum_dIdV,
            Delta           = popt[0],
            T               = T,
            sig0            = sig0,
            Gamma           = Gamma,
            fit_parameter   = ['Delta'],
            popt            = popt,
            pcov            = pcov,
            V_ac_pk         = V_ac_pk,
            V_ac_rms        = V_ac_rms
        )

    @staticmethod
    def fit_T(spectrum_dIdV,Delta=None,sig0 =None,T_guess=0.1,V_ac_pk=None,V_ac_rms=None,Gamma=0,int_point_factor=10,**kwargs):
        """ Fit the dIdV signal with the curve expected for a BCS superconductor. Free parameter: T
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
            Delta (float,optional):             value of the superconducting gap. if None: calculated from the coherence peak positions
            sig0 (float,optional)               normal state conductance. if None: estimated from the data
            T_guess (float,optional):           initial guess for the temeperature in K. default: 0.1K
            V_ac_pk (float,optional):           peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
            V_ac_rms (float,optional):          root mean squared amplitude of the applied AC voltage in V.
            Gamma(float):                       Dynes parameter in eV
            int_point_factor (integer):         perform the DOS convolution on int_point_factor more points than in the spectrum
            kwargs (dictionary):                additional arguments passed to scipy.curvefit
        Returns:
            (popt,pcov,perr,kwargs) where popt = [T], args is a dictionary holding the fit data, to be passed to NIS.real_dIdV(x,**kwargs)
        """
        spectrum = spectrum_dIdV
        if spectrum_dIdV._x_unit[0] != 'V': # not V but maybe mV - convert it to V
            fac = NIS.unit_factors[spectrum_dIdV._x_unit[0]]
            spectrum = deepcopy(spectrum_dIdV)
            spectrum.apply_transformation(lambda x,y: (x*fac,y))

        sig0 = sig0 if sig0 is not None else NIS.get_normal_state_conductance(spectrum)
        Delta = NIS.get_gap_estimate(spectrum) if Delta is None else Delta

        p0 = [ T_guess ]

        if not "p0" in kwargs:
            kwargs["p0"] = p0

        if not "epsfcn" in kwargs:
            kwargs['epsfcn'] = 10e-3

        popt,pcov = spectrum.fit(
            lambda x,T:NIS.real_dIdV(x = x,sig0 = sig0,Delta = Delta,T=T,Gamma=Gamma,
                V_ac_rms=V_ac_rms,V_ac_pk=V_ac_pk,int_point_factor=int_point_factor),
                **kwargs)
        
        return NISFitResult(
            spectrum_dIdV   = spectrum_dIdV,
            Delta           = Delta,
            T               = popt[0],
            sig0            = sig0,
            Gamma           = Gamma,
            fit_parameter   = ['T'],
            popt            = popt,
            pcov            = pcov,
            V_ac_pk         = V_ac_pk,
            V_ac_rms        = V_ac_rms
        )
    
    @staticmethod
    def fit(spectrum_dIdV,Delta_guess=None,sig0_guess =None,T_guess=0.1,V_ac_pk=None,V_ac_rms=None,Gamma_guess=0,int_point_factor=10,x_range=(None,None),**kwargs):
        """ Fit the dIdV signal with the curve expected for a BCS superconductor. Free parameters: Delta, sig0, T, Gamma
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   the dIdV data of the NIS spectrum
            Delta_guess (float,optional):       intitial guess for the value of the superconducting gap. if None: calculated from the coherence peak positions
            sig0_guess (float,optional)         initial guess for the normal state conductance. if None: estimated from the data
            T_guess (float,optional):           initial guess for the temeperature in K. default: 0.1K
            V_ac_pk (float,optional):           peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
            V_ac_rms (float,optional):          root mean squared amplitude of the applied AC voltage in V.
            Gamma_guess(float):                 initial guess for the Dynes parameter in eV
            int_point_factor (integer):         perform the DOS convolution on int_point_factor more points than in the spectrum
            x_range(float,float):               the x-range (in units of x data) of the curve that should be used for fitting.
            kwargs (dictionary):                additional arguments passed to scipy.curvefit
        Returns:
            (popt,pcov,perr,kwargs) where popt = [sig0,Delta,T,Gamma], args is a dictionary holding the fit data, to be passed to NIS.real_dIdV(x,**kwargs)
        """
        spectrum = deepcopy(spectrum_dIdV).crop(x_range)
        if spectrum_dIdV._x_unit[0] != 'V': # not V but maybe mV - convert it to V
            fac = NIS.unit_factors[spectrum_dIdV._x_unit[0]]
            spectrum.apply_transformation(lambda x,y: (x*fac,y))

        sig0_guess = sig0_guess if sig0_guess is not None else NIS.get_normal_state_conductance(spectrum)
        Delta_guess = NIS.get_gap_estimate(spectrum) if Delta_guess is None else Delta_guess

        p0 = [ sig0_guess, Delta_guess, T_guess, Gamma_guess ]

        if not "p0" in kwargs:
            kwargs["p0"] = p0

        if not "epsfcn" in kwargs:
            kwargs['epsfcn'] = 10e-3

        popt,pcov = spectrum_dIdV.fit(
            lambda x,sig0,Delta,T,Gamma:NIS.real_dIdV(x = x,sig0 = sig0,Delta = Delta,T=T,Gamma=Gamma,
                V_ac_rms=V_ac_rms,V_ac_pk=V_ac_pk,int_point_factor=int_point_factor),
                **kwargs)

        return NISFitResult(
            spectrum_dIdV   = spectrum_dIdV,
            Delta           = popt[1],
            T               = popt[2],
            sig0            = popt[0],
            Gamma           = popt[3],
            fit_parameter   = ['sig0','Delta','T','Gamma'],
            popt            = popt,
            pcov            = pcov,
            V_ac_pk         = V_ac_pk,
            V_ac_rms        = V_ac_rms,
            x_range         = x_range
        )
    

class NISFitResult:
    """Class storing the results of an NIS fit"""

    def __init__(self,spectrum_dIdV,Delta,T,sig0,Gamma,fit_parameter,popt,pcov,V_ac_pk=None,V_ac_rms=None,x_range=(None,None)):
        """Create a ISFitResult object
        
        Arguments:
            spectrum_dIdV (simscidapy.Curve):   spectrum which was fitted
            Delta (float):                      value of the superconducting gap
            T (float):                          temperature in K
            sig0 (float)                        normal state conductance
            Gamma(float):                       Dynes parameter in eV
            fit_parameter(list of string):      names of the parameters that were fitted in the order corresponding to the values in popt
            popt (np.array):                    optimum values as calculated by scipy.curvefit
            pcov (np.array):                    covariance matrix as calculated by scipy.curvefit
            V_ac_pk (float,optional):           peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
            V_ac_rms (float,optional):          root mean squared amplitude of the applied AC voltage in V.
            x_range (float,float)               the x-range (in units of x data) of the curve that should be used for fitting.
        """
        self.spectrum = spectrum_dIdV
        self.Delta    = Delta
        self.T = T
        self.sig0 = sig0
        self.Gamma  = Gamma
        self.fit_parameter = fit_parameter
        self.popt = popt
        self.pcov = pcov
        self.x_range = x_range

        if V_ac_pk is None and V_ac_rms is None:
            raise ValueError('Either V_ac_pk or V_ac_rms has to be provided but none was given!')
        self.V_ac_pk = V_ac_pk if V_ac_pk is not None else V_ac_rms*np.sqrt(2)

        self.Delta_err = self.perr[fit_parameter.index('Delta')] if 'Delta' in fit_parameter else 0
        self.T_err = self.perr[fit_parameter.index('T')] if 'T' in fit_parameter else 0
        self.sig0_err = self.perr[fit_parameter.index('sig0')] if 'sig0' in fit_parameter else 0
        self.Gamma_err = self.perr[fit_parameter.index('Gamma')] if 'Gamma' in fit_parameter else 0

    
    @property
    def V_ac_rms(self):
        return self.V_ac_pk / np.sqrt(2)
    
    @V_ac_rms.setter
    def V_ac_rms(self, value):
        self.V_ac_pk = value*np.sqrt(2)
    
    @property
    def perr(self):
        return np.sqrt(np.diag(self.pcov))

    def plot(self,fig=None,ax=None,x=None,x_unit=None,fit_plot_args={},plot_spectrum=True,spectrum_plot_args={},legend='best'):
        """Plot the fit.
        
        Arguments (all arguments are optional):
            fig (matplotlib.figure):            figure to plot into. if fig and ax are None: create a new one
            ax (matplotlib.axes):               axes to plot into. if None: create a new one
            x (np.array(float)):                x values for which to plot the fit in V. if None: use the values of the fit
            x_unit (string):                    unit of the x axis. default: same as the data spectrum             
            fit_plot_args (dictionary):         style arguments for plotting the fit, passed on to axes.plot. If none provided: dashed line
            plot_spectrum (bool):               whether or not to plot the measured data that was fitted in the background, default: True
            spectrum_plot_args (dictionary):    style arguments for plotting the measured data, passed to axes.plot
            legend (string or None):            location of the legend (see matplotlib.axes.plot - loc argument) or None for no legend, default: "best"
        Returns:
            fig,ax                              of the plot
        """
        if x_unit is None:
            x_unit = self.spectrum._x_unit

        if ax is None:
            fig, ax = plt.subplots()
            if plot_spectrum:
                self.spectrum.setup_plot(ax)
            ax.set_xlabel(f'Bias ({x_unit})')
            ax.set_ylabel('dI/dV (a.u.)')
        
        if plot_spectrum:
            if x_unit != self.spectrum._x_unit:
                spec = deepcopy(self.spectrum)
                fac = NIS.unit_factors[self.spectrum._x_unit[0]]/NIS.unit_factors[x_unit[0]]
                spec = spec.apply_transformation(lambda x,y: (x*fac,y))
            else:
                spec = self.spectrum
            spec.plot(ax,plot_args = spectrum_plot_args)
        
        if x is None:
            x = self.spectrum.get_x(self.x_range)*NIS.unit_factors[self.spectrum._x_unit[0]] # x in V
            
        y = NIS.real_dIdV(x,Delta = self.Delta, T = self.T, sig0=self.sig0, V_ac_pk=self.V_ac_pk, Gamma = self.Gamma)

        if fit_plot_args == {}:
            fit_plot_args = dict(ls='--')
        ax.plot(x/NIS.unit_factors[x_unit[0]],y,
            label=f"BCS fit: \nT    = {self.T*1e3:.0f}mK\nΔ    = {self.Delta*1e6:.0f}µeV\nVac = {self.V_ac_rms:.0f}µVrms\nΓ    = {self.Gamma*1e6:.0f}µeV",
            **fit_plot_args)
        if legend is not None:
            ax.legend(loc=legend)
        return fig,ax
    