import numpy as np
from simscidapy import Curve
from Physics import BCS,ddE_Fermi_Dirac_eV,Fermi_Dirac_eV

class STipDeconvolution:
    @staticmethod
    def STip(E,bias,T,Delta,Gamma):
        """Function describing the effect of the superconducting tip on the sample DOS. 
        To be convoluted with the sample DOS to obtain the measured dI/dV.
        Arguments:
            E (np.array(floats)):   Energies at which the function will be evaluated
            bias (float):           Bias applied across the tunnel junction in V
            T (float):              Temperature of the system in K
            Delta (float):          superconducting gap of the tip in eV
            Gamma (float):          Dynes parameter used to avoid diverging features
        Returns:
            np.array(floats)
        """
        Eb = E-bias
        return (-1*BCS.DOS(Eb,Delta,Gamma=Gamma)*ddE_Fermi_Dirac_eV(Eb,T) - (Fermi_Dirac_eV(Eb,T)-Fermi_Dirac_eV(E,T))*BCS.ddE_BCS_DOS(E=Eb,Delta=Delta,Gamma=Gamma))

    @staticmethod
    def _convolution_matrix(f,E,Bias):
        """Generate the convolution matrix for convoluting with f(E,Bias).
        This uses the 'same' mode, where len(f(E,Bias)) = len(E)
        
        Arguments:
            f (y=function(E,Bias)):     function to convolute with. type of y: numpy array with len(y) = len(E)
            E (np.array):               energy values on which to evaluate the function
            Bias (np.array):            bias values shifting the convolution function. len(Bias) = len(E)!. 
        Returns:
            np.array with shape(len(E),len(E))
        """
        l = len(E)
        res = np.zeros([l,l]) # mode = same
        for i,b in enumerate(Bias):
            res[i] = f(E,b)
        return res

    @staticmethod
    def _deconvolution_matrix(f,E,Bias):
        """Generate the deconvolution matrix for deconvoluting a curve that has been convoluted with f(E,Bias)

        Arguments:
            f (y=function(E,Bias)):     function to convolute with. type of y: numpy array with len(y) = len(E)
            E (np.array):               energy values on which to evaluate the function
            Bias (np.array):            bias values shifting the convolution function. len(Bias) = len(E)!. 
        Returns:
            np.array with shape(len(E),len(E))
        """
        M = STipDeconvolution._convolution_matrix(f,E,Bias)
        Minv = np.linalg.pinv(M)
        return Minv

    @staticmethod
    def deconvolution_matrix(E,T,Delta,Gamma):
        """Generate the deconvolution matrix for deconvoluting a dIdV curve measured with a superconducting tip

        Arguments:
            E (np.array):           energy values of the curve
            T (float):              Temperature of the system in K
            Delta (float):          superconducting gap of the tip in eV
            Gamma (float):          Dynes parameter used to avoid diverging features
        Returns:
            np.array with shape(len(E),len(E))
        """
        return STipDeconvolution._deconvolution_matrix(
            lambda e,bias: STipDeconvolution.STip(e,bias,T,Delta,Gamma),
            E,E
        )

    @staticmethod
    def stretch_data(x,y,x_range):
        """Stretch data to a given x-range

        Arguments:
            x (np.array(float)):    x values of the curve to be streched
            y (np.array(float)):    y values of the curve to be streched
            x_range (float,float):  (min,max) energy of the range to which the data will be streched in units of x
        
        Returns:
            (np.array(float),np.array(float)): (x,y) of the streched curve
        """
        x_spacing = x[1]-x[0]
        n_add_left = int(abs(x[0]-x_range[0])/x_spacing)
        n_add_right = int(abs(x_range[1]-x[-1])/x_spacing)
        x_min = x[0]-n_add_left*x_spacing
        x_max = x[-1]+n_add_right*x_spacing
        n_tot = n_add_left + len(x) + n_add_right
        x_new = np.linspace(x_min,x_max,n_tot)
        y_new = np.zeros(x_new.shape)
        y_new[:n_add_left] = y[0]
        y_new[n_add_left:n_add_left+len(x)] = y
        y_new[n_tot-n_add_right:] = y[-1]
        return (x_new,y_new)

    @staticmethod
    def streched_and_reduced_curve(c,x_range,n_pts):
        """Stretch data to a given x-range and reduce the number of x points.
        Useful for limiting the calculation time for high resolution measurements

        Arguments:
            c (simscidapy.Curve):   curve to be streched an reduced
            x_range (float,float):  range to strech the data to
            n_pts (int):            number of points equally spaced in the x_range
        Returns:
            (np.array,np.array):    (x,y) of the streched and reduced curve
        """
        x = np.linspace(*x_range,n_pts)
        return [x,c.evaluate(x,outside_range_value='nearest')] # uses cubic spline interpolation

    @staticmethod
    def deconvolve(deconv_mat,y_data):
        """Deconvolve the curve using the provided deconvolution matrix.

        Arguments:
            deconv_mat (2D np.array):   deconvolution matrix as generated by "deconvolution_matrix".
            y_data (np.array):          y_data of the curve to be deconvoluted. y_data must be defined on the same E values than the deconv_mat!
        """
        return deconv_mat @ y_data