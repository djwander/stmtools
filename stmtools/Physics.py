"""
Natural constants and material properties
all units in SI unless specified differently
"""
import numpy as np
from scipy.optimize import fsolve
from simscidapy import Curve

e =         1.602176634e-19             # C
me =        9.1093837015e-31            # kg: mass of free electron
c =         299792458                   # m / s

kB =        1.38064852e-23              # J/K
kB_eV =     8.617333262145e-5           # eV/K
h =         6.62607015e-34              # J s
h_eV =      4.135667696e-15             # eV s
hbar =      1.054571817e-34             # J s
hbar_eV =   6.582119569e-16             # eV s

L =         2.44e-8                     # W Ohm K^-2 :Lorenz number (for Wiedemann Franz)

### Cu properties
rho_Cu_273 = 15.53e-9 # Ohm m
def rho0_Cu(RRR):
    """ Remaining low-temperature resistance of copper due to impurity scattering for given RRR.
    Arguments: 
            RRR(float): residual resistance ratio
    Returns:
            residual resistance of copper at 0K"""
    return rho_Cu_273/RRR

### 2D vectors
def rot_mat_2D_rad(theta):
    """ generates a numpy 2x2 rotation matrix
    Arguments: 
            theta(float): rotation angle in radian
    Returns:
            np 2x2 rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def rot_mat_2D_deg(theta):
    """ generates a numpy 2x2 rotation matrix
    Arguments: 
            theta(float): rotation angle in degree
    Returns:
            np 2x2 rotation matrix"""
    return rot_mat_2D_rad(np.radians(theta))

### Distribution functions
def Boltzmann_SI(E,T):
    """ Returns the Boltzmann distribution function
    Arguments: 
            E(float or numpy array): energy in J
            T(float or numpy array): temperature in K
    Returns:
            float or numpy array with values of Boltzmann distribution function"""
    return np.exp(-E/(kB*T))

def Boltzmann_eV(E,T):
    """ Returns the Boltzmann distribution function
    Arguments: 
            E(float or numpy array): energy in eV
            T(float or numpy array): temperature in K
    Returns:
            float or numpy array with values of Boltzmann distribution function"""
    return np.exp(-E/(kB_eV*T))

def Fermi_Dirac_SI(E,T,mu=0):
    """ Returns the Fermi-Dirac distribution function
    Arguments: 
            E(float or numpy array): energy in J
            T(float or numpy array): temperature in K
            mu(float or numpy array): chemical potential. default:0
    Returns:
            float or numpy array with values of Fermi-Dirac distribution function"""
    return 1/(np.exp((E-mu)/(kB*T))+1)

def Fermi_Dirac_eV(E,T,mu=0):
    """ Returns the Fermi-Dirac distribution function
    Arguments: 
            E(float or numpy array): energy in eV
            T(float or numpy array): temperature in K
            mu(float or numpy array): chemical potential. default:0
    Returns:
            float or numpy array with values of Fermi-Dirac distribution function"""
    return 1/(np.exp((E-mu)/(kB_eV*T))+1)

def ddE_Fermi_Dirac_eV(E,T):
    """ Returns the d/dE derivative of the Fermi-Dirac distribution function
    Arguments: 
            E(float or numpy array): energy from the Fermi energy in eV 
            T(float or numpy array): temperature in K
    Returns:
            float or numpy array with values of Fermi-Dirac distribution function"""    
    return -1/(kB_eV*T*(np.exp(-E/(kB_eV*T))+2+np.exp(E/(kB_eV*T))))

### Broadening functions
def Gaussian(x,std,x0=0,normed=True):
    """ Gaussian distribution of width std around x0 evaluated at x
    Arguments:
            x (float or numpy array): x values on which to calculate the gaussian distribution
            std (float):              standard deviation of the distribution
            x0 (float):               center of the distribution 
            normed (bool):            True: norm area to 1; False: peak height = 1
    Returns:
            float or numpy array
    """
    norming_fac = 1/(np.sqrt(2*np.pi)*std) if normed else 1
    return np.exp(-(x-x0)**2/(2*std**2))*norming_fac

### Scanning Tunneling Spectroscopy - experimental energy broadening
def apply_lock_in_broadening(spectrum,V_ac_pk=None,V_ac_rms=None,integration_factor=5):
    """ apply the broadening of an AC voltage V_ac (as applied by a lock-in) to 
    the spectral features of a theoretical spectrum
    This corresponds to a convolution of spectrum with 2/(pi*V_ac)*sqrt(V_ac^2-V^2). 
    [see "Fundamental Properties of Molecules on Surfaces" thesis by Nino Hatter 2016 p.13,14]
    Arguments:
        spectrum (Curve):                    theoretical spectrum on which the AC broadening should be applied (x axis in (e)V)
        V_ac_pk (float,optional):            peak amplitude of the applied AC voltage in V. Either V_ac_pk or V_ac_rms has to be provided
        V_ac_rms (float,optional):           root mean squared amplitude of the applied AC voltage in V. 
        integration_factor (int,optional):   how much denser x points should be spaced for the integration
    """
    if V_ac_pk is None and V_ac_rms is None:
        raise ValueError('Either V_ac_pk or V_ac_rms has to be provided but none was specified!')
    V_ac = V_ac_pk if V_ac_pk is not None else np.sqrt(2)*V_ac_rms
    n_pts_R = 256
    x_R = np.arange(-V_ac,V_ac+V_ac/n_pts_R,2*V_ac/n_pts_R)
    R = Curve(x_R,2/(np.pi*V_ac**2)*np.sqrt( np.abs(V_ac**2-x_R**2)))
    spec_x = spectrum.get_x()
    return spectrum.convoluted_with(R,x_resolution=spec_x[1]-spec_x[0],integration_factor=integration_factor)

def apply_temperature_broadening(spectrum,T,integration_factor=5):
    """ apply the broadening by temperature to the spectral features of a theoretical spectrum 
    assuming a normal conducting tip with flat DOS.
    This corresponds to a convolution of spectrum with d/dE f_FD. 
    [see "Fundamental Properties of Molecules on Surfaces" thesis by Nino Hatter 2016 p.14,15]
    Arguments:
        spectrum (Curve):           theoretical spectrum on which the AC broadening should be applied (x axis in (e)V)
        T (float):                  temperature of the tip
        integration_factor (int):   how much denser x points should be spaced for the integration
    """
    n_pts_x = 256
    x = np.arange(-5*kB_eV*T,5*kB_eV*T,10*kB_eV*T/n_pts_x)
    ddE_FD = Curve(x,-1*ddE_Fermi_Dirac_eV(x,T))
    spec_x = spectrum.get_x()
    if 10*kB_eV*T > spec_x[-1]-spec_x[0]:
        raise ValueError(f'The range of the spectrum is not large enough for a Temperature of {T:.2e}K! Check that all your units are SI!')
    return spectrum.convoluted_with(ddE_FD,x_resolution=spec_x[1]-spec_x[0],integration_factor=integration_factor)

### Superconductors
def Maki_DOS(E,Delta,xi):
    """ DOS according to Maki theory - see https://doi.org/10.1063/1.4793793
    E(np array of floats):  energies at which to calculate the DOS in eV
    Delta (float):          gap in eV    
    xi (float):             pair breaking parameter (dimensionless)
    """
    E = E + 1j*np.zeros(E.shape)
    eps = E/Delta
    b = eps**2+xi**2 -1
    f = 108*eps**2*xi**2+2*b**3
    c = f+np.sqrt((f)**2-4*b**6)
    crd = c**(1/3)
    d = b/3+2**(1/3)*b**2/(3*crd) + crd/(3*2**(1/3))
    sqxid = np.sqrt(1-xi**2+d)
    u = 0.5*(np.abs(eps) + sqxid + np.sqrt(1+eps**2-xi**2-d-2*np.abs(eps)*(1+xi**2)/sqxid))
    return np.real(u/np.sqrt(u**2-1))

class BCS:
    """Class grouping all BCS related functions"""
    @staticmethod
    def Delta0_of_Tc(Tc):
        """ Delta at T=0 for a given Tc
        Arguments:
            Tc (float): critical temperature in K
        Returns:
            Delta0 (float): superconducting gap at T=0 in eV
        """
        return 1.76*kB_eV*Tc

    @staticmethod
    def Tc_of_Delta0(Delta0):
        """ Tc for given Delta at T=0
        Arguments:
            Delta0 (float): superconducting gap at T=0 in eV
        Returns:
            Tc (float): critical temperature in K
        """
        return Delta0/(1.76*kB_eV)

    @staticmethod
    def Delta_of_T(T,Tc):
        """ Theoretical temperature dependence of the superconducting gap.
        Arguments:
            T (float or numpy array of floats): temperature at which the gap should be calculated
            Tc (float):                         critical temperature in K
        Returns:
            Delta (float or numpy array of floats): the superconducting gap at the specified temperature(s)
        """
        if type(T) in [int, float,np.float64,np.float32,np.float,np.int,np.int64,np.int32]:
            return BCS.Delta_of_T(np.array([T]),Tc)[0]
        t = T/Tc
        eta = 3
        n_pts = 1000
        def self_consistency(d,t):
            ul = np.sinh(eta)/d
            spacing = ul/n_pts
            x = (np.arange(n_pts)+0.5)*spacing
            sq = np.sqrt(1+x**2)
            y = np.tanh(0.882*d/t*sq)/sq
            int = np.sum(y)*spacing
            return int-eta
        d = np.zeros(T.shape)
        for i,tau in enumerate(t):
            d[i] = fsolve(lambda d: self_consistency(d,tau),x0=0.95)
        return d*BCS.Delta0_of_Tc(Tc)
        
    @staticmethod
    def T_of_Delta(Delta,Delta0):
        """ The inverted function of Delta_of_T. Estimate the temperature of the superconductor from the size of its gap.
        Arguments:
            Delta (float or numpy array of floats): gap value for which the corresponding temperature should be calculated
            Delta0 (float):                         0 temperature value of the gap. Can be calculated from Tc using BCS.Delta0_of_Tc
        Returns:
            T (float or numpy array of floats):     the temperature(s) corresponding to the specified gap(s)
        """
        if type(Delta) in [int,float,np.int,np.int64,np.int32,np.int16,np.float,np.float16,np.float32,np.float64]:
            return BCS.T_of_Delta(np.array([Delta]),Delta0)[0]

        if np.max(Delta) > Delta0:
            raise ValueError(f"{np.max(Delta)} is larger than the gap {Delta0}! This is physically impossible.")

        # calculate Delta of T
        Tc=BCS.Tc_of_Delta0(Delta0)
        T = np.linspace(start=Tc*0.01,stop=Tc*0.997,num=1000)
        Delta_Theory = Curve(T,BCS.Delta_of_T(T,Tc))
        
        res =np.zeros(Delta.shape)
        for i,d in enumerate(Delta):
            res[i] = Delta_Theory.where(d)[0]
        return res       

    @staticmethod
    def DOS(E,Delta,N0=1,Gamma=0):
        """ BCS Density of States
        Arguments:
            E       (float or numpy array): energy from the Fermi level
            Delta   (float): superconducting gap
            N0      (float): density of states in normal state
            Gamma   (float): Dynes parameter
        Returns:
            BCS DOS (float or numpy array of floats): the BCS DOS at the specified energy(s)
        """
        if type(E) in [float,int,np.float64,np.float32]:    
                E = np.array([E]).astype(float)

        res = np.zeros(E.shape)
        if Gamma == 0:
            io = np.argwhere(np.abs(E)>Delta)
            res[io] = N0*np.abs(E[io])/np.sqrt(E[io]**2-Delta**2)
        else:
            io = np.full(len(E),True)
            res[io] = N0*np.abs(np.real((E[io]-1.j*Gamma)/np.sqrt( (E[io]-1.j*Gamma)**2 - Delta**2 ) ))
        return res

    @staticmethod
    def ddE_BCS_DOS(E,Delta,N0=1.0,Gamma=0):
        """ d/dE derivative of the BCS Density of States
        
        Arguments:
            E       (float or numpy array): energy from the Fermi level
            Delta   (float): superconducting gap
            N0      (float): density of states in normal state
            Gamma   (float): Dynes parameter
        Returns:
            (float or numpy array of floats): d/dE of the BCS DOS at the specified energy(s)
        """
        D = Delta
        G = Gamma
        return N0*np.sign(E)*np.real(-D**2/((E-1j*G)**2 - D**2)**(3/2))
    
    @staticmethod
    def dIdV_NIS_eV_symmectric(E_max,E_res,Delta,T_N,int_point_factor=10,Gamma=0):
        """ differential conductance of an NIS junction assuming flat DOS of N
        Arguments:
            E_max   (float): energy from the Fermi level in eV
            E_res   (float): energy resolution in eV
            Delta   (float): superconducting gap in eV
            T_N     (float): temperature of the normal metal
            int_point_factor (integer): perform the integration on int_point_factor more points than E_res
            Gamma   (float): Dynes parameter
        Returns:
            (Curve): dIdV of an NIS junction from -E_max to E_max
        """
        int_limit = 10*kB_eV*T_N
        int_res = E_res/int_point_factor
        E_ddE_FD_pos = np.arange(0,int_limit+int_res,int_res) # ddE_FD is symmetric around 0
        ddE_FD_pos = -1*ddE_Fermi_Dirac_eV(E_ddE_FD_pos,T_N)
        c_ddE_FD = Curve(
                np.concatenate((E_ddE_FD_pos,-1*E_ddE_FD_pos[1:])),
                np.concatenate((ddE_FD_pos,ddE_FD_pos[1:])))
        
        E_DOS_pos = np.arange(Delta-2*int_limit,E_max+int_limit+E_res,int_res)
        DOS_pos = BCS.DOS(E_DOS_pos,Delta,Gamma=Gamma)
        index_Delta = int(2*int_limit/(int_res))+1
        if index_Delta < len(DOS_pos):
            DOS_pos[index_Delta] = BCS.DOS(Delta+0.5*int_res,Delta) # handle the divergence
        c_DOS_pos = Curve(E_DOS_pos,DOS_pos)
        
        res =  c_DOS_pos.convoluted_with(c_ddE_FD,E_res,integration_factor=int_point_factor) # only positive half of the curve
        res.apply_transformation(lambda x,y: (np.concatenate((x,-1*x)),np.concatenate((y,y))))
        res.set_plot_properties({'x_label':'E','x_unit':'eV',
                                 'y_label':'dIdV','y_unit':'a.u.',
                                 'title':f'SIN dIdV at {T_N}K'})
        return res

    @staticmethod
    def dIdV_NIS_eV(E_range,E_res,Delta,T_N,Gamma=0,int_point_factor=10):
        """ differential conductance of a NIS junction assuming flat DOS of N
        Arguments:
            E_range (float,float):          energy range in which to calculate the DOS
            E_res   (float):                energy resolution in eV
            Delta   (float or numpy array): superconducting gap in eV
            T_N     (float or numpy array): temperature of the normal metal
            Gamma   (float):                Dynes parameter
            int_point_factor (integer):     perform the integration on int_point_factor more points than E

        Returns:
            (Curve): dIdV of an NIS junction from E_range[0] to E_range[1]
        """
        E_max = np.max(np.abs(E_range))
        if E_max < Delta-10*kB_eV*T_N:
            raise ValueError(f'The energy range lies far inside the gap, the DOS is zero there! I guess either the given range {E_range}eV or the gap {Delta}eV is wrong.')
        c = BCS.dIdV_NIS_eV_symmectric(E_max,E_res,Delta,T_N,int_point_factor,Gamma)
        return c.cropped(E_range)
