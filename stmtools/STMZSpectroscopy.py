from multiprocessing.sharedctypes import Value
import numpy as np
from NanonisMeasurement import NanZSpectroscopy
from copy import deepcopy
import matplotlib.pyplot as plt
from Physics import hbar,hbar_eV,me, e

class STMZSpectroscopy(NanZSpectroscopy):
    """Class containing standard data treatment applied on I-Z tunneling spectra"""

    distance_unit_factors = {'Tm':1e12,'Gm':1e9,'Mm':1e6,'km':1e3,'m':1,'mm':1e-3,'um':1e-6,'nm':1e-9,'A':1e-10,'Å':1e-10,'pm':1e-12,'am':1e-15}
    current_unit_factors  = {'TA':1e12,'GA':1e9,'MA':1e6,'kA':1e3,'A':1,'mA':1e-3,'uA':1e-6,'nA':1e-9,'pA':1e-12,'aA':1e-15}

    I_noise_floor         = 2e-12 # A

    @staticmethod
    def exp(x,A,z0):
        """Exponential function: A*exp(-x/z0)"""
        return A*np.exp(-x/z0)

    @staticmethod
    def lin(x,a,b):
        """Linear function: a*x + b"""
        return a*x+b

    @staticmethod
    def _SI_log_curve(IZ):
        """Convert the curve to A and m and take the log10 of I"""
        z_fac = STMZSpectroscopy.distance_unit_factors[IZ._x_unit]
        I_fac = STMZSpectroscopy.current_unit_factors[IZ._y_unit]
        return deepcopy(IZ).apply_transformation(lambda z,I: (z*z_fac,np.log10(I*I_fac)))

    @staticmethod
    def find_start_of_noise_floor(IZ):
        """Determine the relative z position at which the current signal falls below the amplifiers noise floor
        
        Arguments:
            IZ (simscidapy.Curve): measured I vs Z curve
        Returns:
            z_NF (float): the z value at which the noise floor is reached
        """
        zs = IZ.where(y=STMZSpectroscopy.I_noise_floor/STMZSpectroscopy.current_unit_factors[IZ._y_unit])
        z = None if len(zs) < 1 else zs[0]
        return z*STMZSpectroscopy.current_unit_factors[IZ._y_unit]
        # IZ_log = STMZSpectroscopy._SI_log_curve(IZ)
        # smooth = IZ_log.smoothed("Gauss",(10,0))
        # d1 = smooth.derivative()
        # d2 = d1.derivative()
        # return d2.get_maximum()[0]

    @staticmethod
    def fit_log_data(IZ,z_range=(None,None),**kwargs):
        """fit the function A exp(-2x/z0) to the data in log representation (-> eponentially decaying uncertainty)

        Arugments:
            IZ (simscidapy.Curve):  measured I vs Z curve
            z_range (float,float):  start and stop height for the fit. if no upper limit given: fit up to the autmatically detected noise floor
            kwargs:                 arguments passed to scipy.curve_fit
        Returns:
            STMZSpectroscopyFitResult
        """
        z_range = list(z_range)
        if z_range[1] is None:
            z_range[1] = STMZSpectroscopy.find_start_of_noise_floor(IZ)
        
        IZ_log = STMZSpectroscopy._SI_log_curve(IZ)

        if not 'p0' in kwargs:
            kwargs['p0'] = [-1e10,-9]
        popt,pcov = IZ_log.fit(STMZSpectroscopy.lin,z_range,**kwargs)
        perr = np.sqrt(np.diag(pcov))
        a = popt[0]
        b = popt[1]
        a_err = perr[0]
        b_err = perr[1]

        z0 = -1*np.log10(np.e)/a
        z0_err = -1*z0/a*a_err
        A = np.power(10,b)
        A_err = np.log(10)*A*b_err
        return STMZSpectroscopyFitResult(
            spectrum= IZ,
            z0 = z0,
            z0_err= z0_err,
            A = A,
            A_err = A_err,
            z_range=z_range
        )

    @staticmethod
    def load_and_fit(filename,fit_z_range=(None,None),z_unit='pm',I_unit='pA',print_result=True):
        """Convenience function loading the data, fitting and plotting it.

        Arguments:
            filename (string):              full filepath to the data file, including the filename 
            z_range (float,float):          start and stop height for the fit. if no upper limit given: fit up to the autmatically detected noise floor
            z_unit (string):                unit to use for the x data. default: 'pm'
            I_unit (string):                unit to use for the y data. default 'pA'
            print_result (bool):            whether or not to print a summary of the fit. default: True
        Returns:
            STMZSpectroscopyFitResult, NanZSpectroscopy, simscidapy.Curve, matplotib.figure, matplotlib.axes:fit, ZS,ZS_IZ,fig,ax """
        
        ZS = NanZSpectroscopy(filename)
        ZS_IZ = ZS.get_curve('Current (A)',x_unit=z_unit,y_unit=I_unit)

        fit = STMZSpectroscopy.fit_log_data(ZS_IZ,z_range=fit_z_range)
        fig,ax = fit.plot()

        if print_result:
            fit.print()
        return fit, ZS,ZS_IZ,fig,ax

    @staticmethod
    def workfunction_rectangular_barrier(z0,workfunction_tip,Vb):
        """Calculate the workfunction of the sample from z0 and the workfunction of the tip.
        Use a rectangular barrier model with a constant height of (Phi_tip+Phi_sample)/2-abs(eVb/2).
        Note: this model is a crude approximation and only holds for small applied bias as well as similar workfunctions.
        
        Arguments:
            z0 (float):                 z0 in m as estimated from the exponential fit
            workfunction_tip (float):   workfunction of the tip material in eV
            Vb (float):                 applied bias in V
        Returns:
            float: workfunction of the sample in eV"""

        barrier = (hbar/z0)**2/(8*me)/e
        return 2*(barrier-np.abs(Vb))-workfunction_tip

    @staticmethod
    def z0_rectangluar_barrier(workfunction_tip,workfunction_sample,Vb):
        """Calculate the current's decay constant from the workfunctions of the tip and sample.
        Use a rectangular barrier model with a constant height of (Phi_tip+Phi_sample)/2-abs(eVb/2).
        Note: this model is a crude approximation and only holds for small applied bias as well as similar workfunctions.
        
        Arguments:
            workfunction_tip (float):       workfunction of the tip material in eV
            workfunction_sample (float):    workfunction of the sample material in eV
            Vb (float):                     applied bias in V
        Returns:
            float: z0 in m"""
        barrier = (workfunction_tip+workfunction_sample)/2-np.abs(Vb/2)
        return 1/(2*np.sqrt(2*me*barrier*e/hbar**2))
        
    
class STMZSpectroscopyFitResult():

    def __init__(self,spectrum, z0,z0_err,A,A_err,z_range):
        self.spectrum   = spectrum
        self.z0         = z0
        self.z0_err     = z0_err
        self.A          = A
        self.A_err      = A_err
        self.z_range    = z_range

    def plot(self,fig=None,ax=None,z=None,z_unit=None,I_unit=None,fit_plot_args={},plot_spectrum=True,spectrum_plot_args={},legend='best'):
        """Plot the fit.
        
        Arguments (all arguments are optional):
            fig (matplotlib.figure):            figure to plot into. if fig and ax are None: create a new one
            ax (matplotlib.axes):               axes to plot into. if None: create a new one
            z (np.array(float)):                z values for which to plot the fit in m. if None: use the values of the fit
            z_unit (string):                    unit of the z axis. default: same as the raw data spectrum             
            I_unit (string):                    unit of the I axis. default: same as the raw data spectrum
            fit_plot_args (dictionary):         style arguments for plotting the fit, passed on to axes.plot. If none provided: dashed line
            plot_spectrum (bool):               whether or not to plot the measured data that was fitted in the background, default: True
            spectrum_plot_args (dictionary):    style arguments for plotting the measured data, passed to axes.plot
            legend (string or None):            location of the legend (see matplotlib.axes.plot - loc argument) or None for no legend, default: "best"
        Returns:
            fig,ax                              of the plot
        """
        if z_unit is None:
            z_unit = self.spectrum._x_unit
        
        if I_unit is None:
            I_unit = self.spectrum._y_unit

        if ax is None:
            fig, ax = plt.subplots()
            if plot_spectrum:
                self.spectrum.setup_plot(ax)
            ax.set_xlabel(f'Z ({z_unit})')
            ax.set_ylabel(f'I ({I_unit})')
            ax.set_yscale('log')

        if plot_spectrum:
            spec = deepcopy(self.spectrum)
            zfac = STMZSpectroscopy.distance_unit_factors[self.spectrum._x_unit]/STMZSpectroscopy.distance_unit_factors[z_unit]
            Ifac = STMZSpectroscopy.current_unit_factors[self.spectrum._y_unit]/STMZSpectroscopy.current_unit_factors[I_unit]
            spec = spec.apply_transformation(lambda x,y: (x*zfac,y*Ifac))
            spec.plot(ax,plot_args = spectrum_plot_args)

        if z is None:
            IZ_z_range = [z/STMZSpectroscopy.distance_unit_factors[self.spectrum._x_unit] if z is not None else z for z in self.z_range]
            z = self.spectrum.get_x(IZ_z_range)*STMZSpectroscopy.distance_unit_factors[self.spectrum._x_unit] # x in m
        
        I = STMZSpectroscopy.exp(z,self.A,self.z0)

        if fit_plot_args == {}:
            fit_plot_args = dict(ls='--')
        ax.plot(z/STMZSpectroscopy.distance_unit_factors[z_unit], I/STMZSpectroscopy.current_unit_factors[I_unit],
            label= f'{self.A*1e12:.0f}pA exp(-z/{self.z0*1e12:.0f}pm)',
            **fit_plot_args)
        if legend is not None:
            ax.legend(loc=legend)
        return fig,ax
    
    def print(self):
        print("result of fitting A*exp(-z/z0)")
        print(f"z0 = {self.z0*1e12:4.0f} +- {self.z0_err*1e12:3.0f} pm")
        print(f'A  = {self.A*1e12:4.0f} +- {self.A_err*1e12:3.0f} pA')
