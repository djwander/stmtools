import numpy as np
from datetime import datetime
import nanonispy as nap
from simscidapy import Curve
from simscidapy import Map
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from pathlib import Path

class NanonisMeasurement:
    """Base class for all types of Nanonis measurements.
        Common properties to all measurements are:
    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """

    unit_factors = {'T':1e12,'G':1e9,'M':1e6,'k':1e3,'m':1e-3,'u':1e-6,'n':1e-9,'p':1e-12,'a':1e-12}

    def __init__(self,filename,title=None,sample='not specified', T=None, area=None,thermometer=None):
        """Create a NanonisMeasurement object
        Arguments: 
            filename (string):              full filepath to the data file, including the filename 
            title (string, optional):       title of the measurement used for labeling the data in plots. if None, use filename
            sample (string,optional):       name of the sample on which the file was measured
            T (float,optional):             measurement temperature in K
            area (int or string,optional):  ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
            thermometer (T=function(datetime)): function taking a datetime timestamp as an argument and returns the corresponding measurement temperature
        """
        self.filename = filename
        self._timestamp = self.load_timestamp()
        self.title = str(title) if title is not None else self.filename.replace('/','\\').split('\\')[-1][:-4]
        self.sample = sample # 
        self.T = T if T is not None else 0
        if thermometer is not None:
            self.T = thermometer(self.timestamp)
        self.area = area        # area: ID of the area where the measurement was executed in (area changes when doing x or y coarse motion)
    
    @property
    def filename(self):
        return str(self._filename)

    @filename.setter
    def filename(self,filename):
        path = Path(filename)
        if not path.is_file():
            raise ValueError(f'Error when reading "{filename}": this is not a valid filename!')
        self._filename = path

    @property
    def timestamp(self):
        return self._timestamp
        
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self,T):
        try:
            self._T = float(T)
        except:
            raise ValueError(f'"{T}" can not be converted to a float! Give a valid number (in K)')

    def load_timestamp(self):
        raise NotImplementedError("load_timestamp not implemented!")


class NanSpectroscopy(NanonisMeasurement):
    """Base class for all types Nanonis Spectroscopy measurements.

    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        Vb (float):             Bias voltage in V

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """

    get_shortcuts = {   'V_start':'Bias Spectroscopy>Sweep Start (V)',
                        'V_stop':'Bias Spectroscopy>Sweep End (V)',
                        't_settling': 'Bias Spectroscopy>Settling time (s)',
                        't_int': 'Bias Spectroscopy>Integration time (s)',
                        'num_pt': 'Bias Spectroscopy>Num Pixel',
                        'num_sweeps': 'Bias Spectroscopy>Number of sweeps',
                        'I_sp': 'Current>Current (A)',
                        'A_sp': 'Oscillation Control>Amplitude Setpoint (m)',
                        'f_sp': 'Oscillation Control>FrequencyShift (Hz)'
                        }

    def __init__(self,filename,title=None,sample=None, T=None, area=None, grid=None,thermometer=None):
        """Create a NanonisSpectroscopy object
        Arguments: 
            filename (string):              full filepath to the data file, including the filename 
            title (string, optional):       title of the measurement used for labeling the data in plots. if None, use filename
            sample (string,optional):       name of the sample on which the file was measured
            T (float,optional):             measurement temperature in K
            area (int or string,optional):  ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
            grid ([NanGridSpectroscopy,index_x,index_y):     if the spectrum was measured as part of a grid: grid where the spectrum got extracted from, index_x: its x index in the grid, index_y its y index in the grid
            thermometer (T=function(datetime)): function taking a datetime timestamp as an argument and returns the corresponding measurement temperature
        """
        if grid is not None:
            self.grid = grid[0]
            self.index_x = grid[1]
            self.index_y = grid[2]
        else:
            self.grid = None
        super().__init__(filename,title=title,sample=sample,T=T, area=area,thermometer=thermometer)
        self.spectra = {} # list of all available signals, each signal is a list of curve objects
        self.meta = {} # dictionary containing the metadata of the measurement
        self.load()

    @property
    def Vb(self):
        return float(self.get('Bias>Bias (V)'))
    
    def load_timestamp(self):
        """Load the timestamp of the measurement from the data file.

        Returns:
            datetime: timestamp of the measurement
        """
        if self.grid is None:
            tmp = nap.read.Spec(self.filename)
            ts = datetime.strptime(tmp.header['Date'], "%d.%m.%Y %H:%M:%S")
            del tmp
            return ts
        else:
            return self.grid.get_timestamp_spec(self.index_x,self.index_y)

    def load(self):
        """Load the data from the data file into memory."""
        if self.grid is not None: # in case the spectrum is part of a grid, the NanGridSpectroscopy class will take care of providing the data
            return 
        data = nap.read.Spec(self.filename)
        self.meta = data.header

        self.spectra={} # empty data if it was already loaded
        for _, (k, v) in enumerate(data.signals.items()):
            if '[bwd]' in k:
                scan_dir = 'bwd'
            else:
                scan_dir = 'fwd'
            k=k.replace(' [bwd]','')
            filtered = '[filt]' in k
            k = k.replace(' [filt]','')
            if '[' in k: # if still brackets left: its the number of the scan. remove it.
                k=k[:k.index('[')]+k[k.index(']')+2:]
            if filtered:
                k = k + ' [filt]'
            if not k in self.spectra:
                self.spectra[k] = {'fwd':[],'bwd':[]}
            self.spectra[k][scan_dir].append(v)
        return self

    def get_curve(self,signal_x,signal_y,scan_dir="avg",index=None,x_unit=None,y_unit=None):
        """Create a curve object containing the data for the given signal.
        
        Arguments:
            signal_x (string): name of the signal used for the x-axis e.g. "Bias calc (V)"
            signal_y (string): name of the signal e.g. "Current (A)"
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
            x_unit (string): unit to use for the x data (e.g. 'mV')
            y_unit (string): unit to use for the y data

        Returns:
            Curve: curve object containing the data for the given signal
        
        Raises:
            ValueError: when x_unit or y_unit are not compatible with the units of the data (e.g. x_unit='mV' but the data is 'm')
        """
        y_data = np.zeros(len(self.spectra[signal_x]['fwd'][0]))
        ct = 0
        indices = []
        if type(index) is int:
            indices.append(index)
        elif type(index) is list:
            indices = index
        elif index is None:
            dir = scan_dir if scan_dir in ["fwd","bwd"] else "fwd"
            indices = range(len(self.spectra[signal_y][dir]))
        
        for i in indices:
            if (scan_dir == "fwd" or scan_dir == "avg") and i < len(self.spectra[signal_y]["fwd"]):
                y_data += self.spectra[signal_y]["fwd"][i]
                ct += 1
            if (scan_dir == "bwd" or scan_dir == "avg") and i < len(self.spectra[signal_y]["bwd"]):
                y_data += self.spectra[signal_y]["bwd"][i]
                ct += 1
        y_data /= ct
        dx_label = signal_x[:signal_x.index('(')-1]
        dx_unit = signal_x[signal_x.index('(')+1:signal_x.index(')')]
        dy_label = signal_y[:signal_y.index('(')-1]
        dy_unit = signal_y[signal_y.index('(')+1:signal_y.index(')')]
        c = Curve(x=self.spectra[signal_x]['fwd'][0],y=y_data,x_label=dx_label,x_unit=dx_unit,title=self.title,y_label=dy_label,y_unit=dy_unit)
        if x_unit is not None:
            if x_unit[1:] != dx_unit:
                raise ValueError(f'The asked unit "{x_unit}" is not compatible with the data unit "{dx_unit}"!')
            else:
                c.apply_transformation(lambda x,y: (x/NanonisMeasurement.unit_factors[x_unit[0]],y))
                c.set_plot_properties({'x_unit':x_unit})
        if y_unit is not None:
            if y_unit[1:] != dy_unit:
                raise ValueError(f'The asked unit "{y_unit}" is not compatible with the data unit "{dy_unit}"!')
            else:
                c.apply_transformation(lambda x,y: (x,y/NanonisMeasurement.unit_factors[y_unit[0]]))
                c.set_plot_properties({'y_unit':y_unit})
        return c
    
    def get_available_signals(self):
        """Get a list of all available signals in the data file.
        
        Returns:
            list of strings: containing the names of the available signals"""
        return self.spectra.keys()

    def get_available_properties(self):
        """Get all available properties in the measurement's meta data.
        
        Returns:
            list of strings: containing the available properties"""
        return self.meta.keys()

    def get(self,property):
        """Get an arbitrary property of the measurement's meta data.
        
        Returns:
            string: the value of the property
        
        Raises:
            ValueError: if the property does not exist"""
        if property in self.get_shortcuts:
            property = self.get_shortcuts[property]
        if not property in self.meta.keys():
            raise ValueError(f"No property '{property}' in {self.filename}!")
        return self.meta[property]

    def get_position(self):
        """Get the real space tip position at which the measurement was performed.
        
        Returns:
            (float,float): (x,y) position of the measurement in m"""
        return (float(self.get('X (m)')),float(self.get('Y (m)')))

    def plot_position(self,ax,args={'color':'r','marker':'.'},annotate=True):
        """plot a dot at the position where the spectrum was taken (X,Y in meter)
        
        Arguments:
            ax (matplotlib.axes):       axes to plot the position into
            args (dict):                arguments passed to matplotlib.axes.plot
            annotate (bool):            whether or not the measurements title should be written next to the point
        """
        ax.plot(*self.get_position(),**args)
        if annotate:
            self.annotate(ax,text=self.title)

    def annotate(self,ax,text,xytext=(5,5),textcoords='offset pixels',annotation_clip=None,text_args={}):
        """plot an annotation on ax at the position at which the spectrum was taken
        
        Arguments:
              ax (mpl axes): axes to plot into
              text (string): text to show
              xytext (float,float): position of the text, default: pixel relative to the point plotted by plot_position
              textcoords (string): coordinates used to specify the position of the text.
              annotation_clip (bool or None): Whether to draw the annotation when the annotation point xy is outside the axes area
              args (dict): dictionary with additional arguments passed to annotate or text
        for further information about additional arguments see:
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.annotate.html and
        https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text
        """
        xy = self.get_position()
        ax.annotate(text,xy,xytext,'data',textcoords,annotation_clip,**text_args)


class NanBiasSpectroscopy(NanSpectroscopy):
    """Class for Bias Spectroscopy measurements.

    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        Vb (float):             Bias voltage in V
        Isp (float):            Current setpoint in A

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """
    def __init__(self,filename,title=None,sample='not specified', T=None, area=None, grid=None, thermometer=None):
        """Create a NanBiasSpectroscopy object

        Arguments: 
            filename (string):              full filepath to the data file, including the filename 
            title (string, optional):       title of the measurement used for labeling the data in plots. if None, use filename
            sample (string,optional):       name of the sample on which the file was measured
            T (float,optional):             measurement temperature in K
            area (int or string,optional):  ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
            grid ([NanGridSpectroscopy,index_x,index_y):     if the spectrum was measured as part of a grid: grid where the spectrum got extracted from, index_x: its x index in the grid, index_y its y index in the grid
            ,thermometer=None
        """
        super().__init__(filename=filename,title=title,sample=sample, T=T, area=area,grid=grid,thermometer=thermometer)
    
    def get_curve(self,signal,scan_dir="avg",index=None,x_unit=None,y_unit=None):
        """Create a curve object containing the data for the given signal.
        
        Arguments:
            signal (string): name of the signal e.g. "Current (A)"
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
            x_unit (string): unit to use for the x data (e.g. 'mV')
            y_unit (string): unit to use for the y data

        Returns:
            Curve: curve object containing the data for the given signal
        
        Raises:
            ValueError: when x_unit or y_unit are not compatible with the units of the data (e.g. x_unit='mV' but the data is 'm')
        """
        ret = super().get_curve("Bias calc (V)",signal,scan_dir,index,x_unit,y_unit)
        ret.set_plot_properties({'x_label':'Bias'})
        return ret
    
    @property
    def Isp(self):
        # 1st: try to get it from Z-Controller setpoint
        if 'Z-Controller>Setpoint unit' in self.meta:
            if self.meta['Z-Controller>Setpoint unit'] == 'A':
                return float(self.meta['Z-Controller>Setpoint'])
        # if that does not work, extract it from the start of the scan. 
        # Note: this assumes that the starting bias is the same as the bias used for fixing the height
        cav = self.get_curve('Current (A)')
        x = float(self.get('Bias Spectroscopy>Sweep Start (V)'))
        Isp = cav.evaluate(x)
        print('Bias Spectroscopy: Isp extracted from spectrum. Take care, this might be a wrong value!')
        return Isp

    def get_Rt(self):
        """Get the tunneling resistance in Ohm.
        
        Returns:
            float: tunneling resistance in Ohm"""
        return self.Vb/self.Isp

    def get_Sigmat(self):
        """Get the tunneling conductance in Siemens.
        
        Returns:
            float: tunneling conductance in Siemens"""
        return self.Isp/self.Vb
        
    def get_IV(self,scan_dir="avg",index=None,x_unit=None,y_unit=None):
        """Create a curve object containing the IV curve.
        
        Arguments:
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
            x_unit (string): unit to use for the x data (e.g. 'mV')
            y_unit (string): unit to use for the y data

        Returns:
            Curve: curve object containing the IV data
        
        Raises:
            ValueError: when x_unit or y_unit are not compatible with the units of the data (e.g. x_unit='mV' but the data is 'm')
        """
        return self.get_curve("Current (A)",scan_dir,index,x_unit,y_unit)

    def get_dIdV(self,lock_in_signal='Input 3 (V)',scan_dir='avg',index=None,dig_dIdV_Gauss_kernel_size=3,x_unit=None,y_unit=None):
        """Create a curve object containing the dI/dV data.
        
        Arguments:
            lock_in_signal (string): signal name used to record the lock-in signal; if None: calculate dIdV by derivation of IV
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
            dig_dIdV_Gauss_kernel_size: if dIdV is calculated from IV (lock_in_signal=None): smooth the IV with a Gaussian kernel of a width of dig_dIdV_Gauss_kernel_size data points
            x_unit (string): unit to use for the x data (e.g. 'mV')
            y_unit (string): unit to use for the y data

        Returns:
            Curve: curve object containing the data for the given signal
        
        Raises:
            ValueError: when x_unit or y_unit are not compatible with the units of the data (e.g. x_unit='mV' but the data is 'm')
        """
        if lock_in_signal is not None:
            try:
                c =  self.get_curve(lock_in_signal,scan_dir,index,x_unit,y_unit)
                c.set_plot_properties({'y_label':'dI/dV','y_unit':'a.u.','x_label':'Bias'})
                return c
            except KeyError:
                raise ValueError(f"Error when loading dIdV.\n{self.filename} does not contain the signal '{lock_in_signal}'!")
        else:
            IV = self.get_IV(scan_dir,index)
            IV.smoothen('Gauss',(dig_dIdV_Gauss_kernel_size,0))
            dIdV = IV.derivative()
            dIdV.set_plot_properties({'y_label':'dI/dV','y_unit':'a.u.','x_label':'Bias','title':'numerically calulated dIdV'})
            return dIdV
    
    def correct_bias_offset(self, scan_dir='avg',index=None):
        """Shift the bias such that 0 Bias = 0 Current,
        using IV data with given scan_dir and index.
        Assumes that there is no gap at 0 bias in DOS
        
        Arguments:
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
        """
        IV = self.get_IV(scan_dir,index)
        x0 = IV.where(0,1e-7,interpolation='spline',interpolation_args={'k':1})
        x_sub = x0[np.argmin(np.abs(x0))]
        self.spectra['Bias calc (V)']['fwd'][0] -= x_sub


class NanZSpectroscopy(NanSpectroscopy):
    """Class for Bias Spectroscopy measurements.

    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        Vb (float):             Bias voltage in V

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """
    def __init__(self,filename,title=None,sample='not specified', T=None, area=None, grid=None,thermometer=None):
        """Create a NanBiasSpectroscopy object

        Arguments: 
            filename (string):              full filepath to the data file, including the filename 
            title (string, optional):       title of the measurement used for labeling the data in plots. if None, use filename
            sample (string,optional):       name of the sample on which the file was measured
            T (float,optional):             measurement temperature in K
            area (int or string,optional):  ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
            grid ([NanGridSpectroscopy,index_x,index_y):     if the spectrum was measured as part of a grid: grid where the spectrum got extracted from, index_x: its x index in the grid, index_y its y index in the grid
            thermometer (T=function(datetime)): function taking a datetime timestamp as an argument and returns the corresponding measurement temperature
        """
        super().__init__(filename=filename,title=title,sample=sample,T=T,area=area,grid=grid,thermometer=thermometer)
    
    def get_curve(self,signal,scan_dir="avg",index=None,x_unit=None,y_unit=None):
        """Create a curve object containing the data for the given signal.
        
        Arguments:
            signal (string): name of the signal e.g. "Current (A)"
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
            x_unit (string): unit to use for the x data (e.g. 'pm')
            y_unit (string): unit to use for the y data

        Returns:
            Curve: curve object containing the data for the given signal
        
        Raises:
            ValueError: when x_unit or y_unit are not compatible with the units of the data (e.g. x_unit='mV' but the data is 'm')
        """
        return super().get_curve("Z rel (m)",signal,scan_dir,index,x_unit,y_unit)


class NanFrequencySweep(NanSpectroscopy):
    """Class for Frequency Sweep measurements.

    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        Vb (float):             Bias voltage in V

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """
    def __init__(self,filename,title=None,sample='not specified', T=None, area=None, grid=None,thermometer=None):
        """Create a NanFrequencySweep object

        Arguments: 
            filename (string):              full filepath to the data file, including the filename 
            title (string, optional):       title of the measurement used for labeling the data in plots. if None, use filename
            sample (string,optional):       name of the sample on which the file was measured
            T (float,optional):             measurement temperature in K
            area (int or string,optional):  ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
            grid ([NanGridSpectroscopy,index_x,index_y):     if the spectrum was measured as part of a grid: grid where the spectrum got extracted from, index_x: its x index in the grid, index_y its y index in the grid
            thermometer (T=function(datetime)): function taking a datetime timestamp as an argument and returns the corresponding measurement temperature
        """
        super().__init__(filename=filename,title=title,sample=sample,T=T,area=area,grid=grid,thermometer=thermometer)
    
    def get_center_frequency(self):
        """Get the center frequency used for the sweep
        
        Returns:
            float: the center frequency of the sweep in Hz
        """
        return super().get_curve('Frequency Shift (Hz)','Center Frequency (Hz)','avg').mean()
        
    def get_curve(self,signal,scan_dir="avg",index=None,x_unit=None,y_unit=None):
        """Create a curve object containing the data for the given signal.
        
        Arguments:
            signal (string): name of the signal e.g. "Current (A)"
            scan_dir (string): possible values:
                "fwd": forward scan only
                "bwd": backward scan only
                "avg": (default) pointwise mean of forward and backward scan
            index (int,list of int,None): 
                int: index of the scan to be returned if several available. 
                list of int: list of the indices of scans to be averaged
                None: average of all available spectra
            x_unit (string): unit to use for the x data (e.g. 'mV')
            y_unit (string): unit to use for the y data

        Returns:
            Curve: curve object containing the data for the given signal
        
        Raises:
            ValueError: when x_unit or y_unit are not compatible with the units of the data (e.g. x_unit='mV' but the data is 'm')
        """
        c = super().get_curve('Frequency Shift (Hz)',signal,scan_dir,index,x_unit=x_unit,y_unit=y_unit)
        xfac = 1 if x_unit is None else NanonisMeasurement.unit_factors[x_unit[0]]
        c.apply_transformation(lambda x,y: (x+self.get_center_frequency()/xfac,y))
        c.set_plot_properties({'x_label':'Frequency'})
        return c


class NanHistory(NanonisMeasurement):
    """Class for History measurements. Useful for noise analysis.

    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        Vb (float):             Bias voltage in V

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """

    def __init__(self,filename,title=None,sample='not specified', T=None, area=None,thermometer=None):
        """ Nanonis History recording.
        Arguments:
            filename (string)
            sample (string):    Name of the sample
            T (float):          Temperature of the experiment
            area:               ID of the area where the experiment was performed (optional)
            thermometer (T=function(datetime)): function taking a datetime timestamp as an argument and returns the corresponding measurement temperature
        """
        super().__init__(filename,title=title,sample=sample,T=T, area=area,thermometer=thermometer)
        self.data = nap.read.Spec(self.filename)
        

    def load_timestamp(self):
        """Load the timestamp of the measurement from the data file.

        Returns:
            datetime: timestamp of the measurement
        """
        data = nap.read.Spec(self.filename)
        return datetime.strptime(data.header['Date'], "%d.%m.%Y %H:%M:%S")

    def get_curve(self,signal,x_unit=None,y_unit=None):
        """ return a curve object containing the data for the given signal
        Agruments:
            signal (string): name of the signal e.g. "Current (A)"
        """
        y = self.data.signals[signal]
        if "Time (s)" in self.data.signals: # Oscilloscopy kind of measurement
            t = self.data.signals['Time (s)']
        else:                               # History kind of measurement
            t_tot = float(self.data.header['Sample Period (ms)'])
            t = np.linspace(0,t_tot/1000,len(y))
        
        label = signal[:signal.index('(')-1]
        unit = signal[signal.index('(')+1:signal.index(')')]
        c = Curve(x=t,y=y,y_label=label,y_unit=unit,x_label='Time',x_unit=('s'),title=self.title)
        if x_unit is not None:
            if x_unit[1:] != 's':
                raise ValueError(f'The asked unit "{x_unit}" is not compatible with the data unit "s"!')
            else:
                c.apply_transformation(lambda x,y: (x/NanonisMeasurement.unit_factors[x_unit[0]],y))
                c.set_plot_properties({'x_unit':x_unit})
        if y_unit is not None:
            if y_unit[1:] != unit:
                raise ValueError(f'The asked unit "{y_unit}" is not compatible with the data unit "{unit}"!')
            else:
                c.apply_transformation(lambda x,y: (x,y/NanonisMeasurement.unit_factors[y_unit[0]]))
                c.set_plot_properties({'y_unit':y_unit})
        return c

    def get_available_signals(self):
        """ get a list of all signals names available for this recording
        Returns:
            list of strings
        """
        return self.data.signals.keys()


class NanGridSpectroscopy(NanonisMeasurement):
    """Class for Grid (Bias) Spectroscopy measurements.

    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        Vb (float):             Bias voltage in V
        Isp (float):            Current setpoint in A

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """

    def __init__(self,filename,title=None,sample='not specified', T=None, area=None, Isp = None, Vb=None,thermometer=None):
        """Create a NanGridSpectroscopy object

        Arguments: 
            filename (string):              full filepath to the data file, including the filename 
            title (string, optional):       title of the measurement used for labeling the data in plots. if None, use filename
            sample (string,optional):       name of the sample on which the file was measured
            T (float,optional):             measurement temperature in K
            area (int or string,optional):  ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
            thermometer (T=function(datetime)): function taking a datetime timestamp as an argument and returns the corresponding measurement temperature
        """
        self.data = None
        super().__init__(filename,title,sample,T,area,thermometer=thermometer)
        self._thermometer = thermometer
        self.load()

        self.center_x = self.data.header["pos_xy"][0]
        self.center_y = self.data.header["pos_xy"][1]
        self.angle = -self.data.header["angle"] # Nanonis angles are clockwise..
        self.size_x = self.data.header["size_xy"][0]
        self.size_y = self.data.header["size_xy"][1]
        self.pts_x = self.data.header["dim_px"][0]
        self.pts_y = self.data.header["dim_px"][1]
        
        self.topo_min = np.min(self.data.signals['topo'])
        self.topo_max = np.max(self.data.signals['topo'])
        self._calculate_helper_vectors()        

        try:
            self.Isp = Isp if Isp is not None else self.data.signals['params'][0][0][self.data.header['experimental_parameters'].index('Scan:Current (A)')+len(self.data.header['fixed_parameters'])]
        except:
            raise ValueError('Could not read Isp from the file directly. Please provide it via the Isp parameter.')
        self.Vb = Vb if Vb is not None else self.data.signals['params'][0][0][0] # take Vb if given, otherwise 
    
    def load(self):
        """Load the data from the data file into memory."""
        if self.data is None:
            self.data = nap.read.Grid(self.filename)

    def load_timestamp(self):
        """Load the timestamp of the measurement from the data file.

        Returns:
            datetime: timestamp of the measurement
        """
        self.load()
        return datetime.strptime(self.data.header['start_time'], "%d.%m.%Y %H:%M:%S")

    def scale(self,factor_x,factor_y):
        """Scale the size of the grid.
        
        Arguments:
            factor_x (float): factor by which to scale the x-axis
            factor_y (float): factor by which to scale the y-axis
        """
        self.center_x *= factor_x
        self.center_y *= factor_y
        self.size_x *= factor_x
        self.size_y *= factor_y
        self._calculate_helper_vectors()

    def _calculate_helper_vectors(self):
        # calculate some helper vectors
        self.vec_center = np.array([self.center_x,self.center_y])
        
        self.angle_rad = self.angle*2*np.pi/360
        self.vec_x_tot = self.size_x*np.array([np.cos(self.angle_rad),np.sin(self.angle_rad)])
        self.vec_y_tot = self.size_y*np.array([-np.sin(self.angle_rad),np.cos(self.angle_rad)])
        self.vec_x_step = self.vec_x_tot/self.pts_x
        self.vec_y_step = self.vec_y_tot/self.pts_y

    def get_available_signals(self):
        """Get a list of all available signals in the data file.
        
        Returns:
            list of strings: containing the names of the available signals"""
        l = list(self.data.signals.keys())
        l.remove('params')
        return l

    @staticmethod
    def nop():
        pass

    def get_spectrum(self,index_x,index_y):
        """Get an individual spectrum of the grid.

        Arguments:
            index_x (int): x index of the spectrum in the grid
            index_y (int): y index of the spectrum in the grid

        Returns:
            NanBiasSpectroscopy: the spectrum at the indicated position
        """
        title = self.title + f'(X={index_x},Y={index_y})'
        spec = NanBiasSpectroscopy(self.filename,title,self.sample,self.T,self.area,[self,index_x,index_y],thermometer=self._thermometer)
        
        spec.load = self.nop # disable loading because it would fail

        spec.meta = deepcopy(self.data.header)
        # load meta data
        for i,v in enumerate(self.data.header['fixed_parameters']+self.data.header['experimental_parameters']):
            spec.meta[v] = self.data.signals['params'][index_y][index_x][i]
        # build spectra
        spec.spectra={} # empty data if it was already loaded
        for _, (k, v) in enumerate(self.data.signals.items()):
            if k == 'sweep_signal':
                continue
            if '[bwd]' in k:
                scan_dir = 'bwd'
            else:
                scan_dir = 'fwd'
            k=k.replace(' [bwd]','')
            filtered = '[filt]' in k
            k = k.replace(' [filt]','')
            if '[' in k: # if still brackets left: its the number of the scan. remove it.
                k=k[:k.index('[')]+k[k.index(']')+2:]
            if filtered:
                k = k + ' [filt]'
            if not k in spec.spectra:
                spec.spectra[k] = {'fwd':[],'bwd':[]}
            spec.spectra[k][scan_dir].append(v[index_y][index_x])

        # add 'Bias calc (V)' signal
        spec.spectra['Bias calc (V)'] = {'fwd':[self.data.signals['sweep_signal']]}

        return spec

    def number_of_valid_spectra(self):
        """Number of valid spectra in the grid.
        If the measurement was terminated early, the not measured spectra contain just zeros.

        Returns:
            int: number of valid spectra in the grid
        """
        invalid_spectra = np.argwhere(self.data.signals[self.get_available_signals()[0]]==0)
        if len(invalid_spectra) > 0:
            first_zero = invalid_spectra[0]
            return first_zero[0]*self.pts_x + first_zero[1]
        else:
            return self.pts_x*self.pts_y

    @property
    def start_time(self):
        return datetime.strptime(self.data.header['start_time'], "%d.%m.%Y %H:%M:%S")
    
    @property
    def end_time(self):
        return datetime.strptime(self.data.header['end_time'], "%d.%m.%Y %H:%M:%S")

    def time_per_spectrum(self):
        """Average time needed to measure one spectrum.

        Returns:
            float: time per spectrum in s
        """
        spectime=(self.end_time-self.start_time)/self.number_of_valid_spectra()
        return spectime

    def get_timestamp_spec(self,index_x,index_y):
        """Get timestamp for a specific spectrum
        
        Arguments:
            index_x (int): x index of the spectrum
            index_y (int): y index if the spectrum

        Returns:
            datetime: timestamp for the specific spectrum
        """
        ts = self.start_time + self.time_per_spectrum() * (index_y*self.pts_x+index_x)
        return ts

    def draw_surrounding_box(self,ax,surrounding_box_args={"linewidth":1,"color":"k"}):
        """Draw a rectangle surrounding the area of the grid.

        Arguments:
            ax (matplotlib.axes):           axes to draw the rectangle into
            surrounding_box_args (dict):    style arguments for the rectangle passed to axes.plot
        """
        corner_points = self.vec_center+np.array([
                         -self.vec_x_tot/2-self.vec_y_tot/2,
                         self.vec_x_tot/2-self.vec_y_tot/2,
                         self.vec_x_tot/2+self.vec_y_tot/2,
                         -self.vec_x_tot/2+self.vec_y_tot/2,
                         -self.vec_x_tot/2-self.vec_y_tot/2])
        x_pts = [cp[0] for cp in corner_points]
        y_pts = [cp[1] for cp in corner_points]
        ax.plot(x_pts,y_pts,**surrounding_box_args)
        
    def draw_grid_lines(self,ax,grid_line_args={"linewidth":1,"linestyle":"--","color":"gray"}):
        """Draw a line for each row and each column of the grid.
        The spectra have been measured at the intersecting points.

        Arguments:
            ax (matplotlib.axes):     axes to draw into
            grid_line_args (dict):    style arguments for the lines passed to axes.plot
        """
        # horizontal lines
        start = self.vec_center-self.vec_x_tot/2-self.vec_y_tot/2+self.vec_y_step/2
        stop = self.vec_center+self.vec_x_tot/2-self.vec_y_tot/2 +self.vec_y_step/2
        for _ in range(self.pts_y):
            ax.plot([start[0],stop[0]],[start[1],stop[1]],**grid_line_args)
            start = start + self.vec_y_step
            stop = stop + self.vec_y_step
        # vertical lines
        start = self.vec_center-self.vec_x_tot/2-self.vec_y_tot/2+self.vec_x_step/2
        stop = self.vec_center-self.vec_x_tot/2+self.vec_y_tot/2 +self.vec_x_step/2
        for _ in range(self.pts_x):
            ax.plot([start[0],stop[0]],[start[1],stop[1]],**grid_line_args)
            start = start + self.vec_x_step
            stop = stop + self.vec_x_step

    
    def draw_points(self,ax,default_pt_args={"marker":".","color":"b"},pt_style_callback=None,
                    color_map="afmhot",show_colorbar=True,ax_colorbar = None,colorbar_range=(None,None),colorbar_label=''):
        """Plot points at the positions of the individual spectra of the grid.

        Arguments:
            ax (matplotlib.axes):                           axes to draw into
            default_pt_args:                                default arguments for the style of the points, used if pt_style_callback is None or returns None
            pt_style_callback(Nan_BiasSpectroscopy,x,y):    function that returns a plot arguments dictionary for the point x,y. First argument: spectrum measured at that point; 
                                                            if None: default pt_style is used for all points: if the dictionary contains the key "value", the color of the point is calculated according to the color map
            color_map:                                      color map used to calculate the color of all points that have a key "value"
        """
        
        # 1st get all styles / values
        style = []
        for y in range(self.pts_y):
            col_style = []
            for x in range(self.pts_x):
                args = default_pt_args
                if pt_style_callback is not None:
                    args = pt_style_callback(self.get_spectrum(x,y),x,y)
                col_style.append(args)
            style.append(col_style)
        
        values = np.array([s['value'] for y in style for s in y if 'value' in s.keys() ])
        if len(values) > 0:
            max = np.max(values) if colorbar_range[1] is None else colorbar_range[1]
            min = np.min(values) if colorbar_range[0] is None else colorbar_range[0]
            cmap = plt.get_cmap(color_map)
            if max < min:
                tmp = max
                max = min
                min = tmp
            norm = Normalize(min,max,clip=True)
            scmap = ScalarMappable(norm,cmap)
        for iy, y in enumerate(style):
            for ix, s in enumerate(y):
                pos = (self.vec_center-self.vec_x_tot/2-self.vec_y_tot/2+
                    self.vec_x_step*(ix+0.5)+self.vec_y_step*(iy+0.5))
                if 'value' in s:
                    s['color'] = scmap.to_rgba(s['value'])
                    del s['value']
                ax.plot(*pos,**s)
        if show_colorbar and len(values) > 0:
            if ax_colorbar is None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
            else:
                cax = ax_colorbar
            scmap.set_array([])
            plt.colorbar(scmap,cax=cax)
            cax.set_ylabel(colorbar_label)
        
    def plot(self, ax, draw_surrounding_box=True,surrounding_box_args={"linewidth":1,"color":"k"},
                draw_grid_lines=True,grid_line_args={"linewidth":1,"linestyle":"--","color":"gray"},
                draw_points=True,default_pt_args={"marker":".","color":"b"},pt_style_callback=None,color_map="afmhot",show_colorbar=True,ax_colorbar=None,colorbar_range=(None,None),colorbar_label='',
                annotate=True):
        """Plot the real space position of the grid.
            
            Arguments:
                draw_surrounding_box: whether or not the surrounding box should be drawn
                surrounding_box_args: plot arguments for changing the line style of the surrounding box
                draw_grid_lines: whether or not a grid of lines should be drawn
                grid_line_args: plot arguments for chaning the line style of the grid lines
                draw_points:   whether or not points should be drawn at each spectrum position
                default_pt_args: default arguments for the style of the points
                pt_style_callback(grid_object,x,y): function that returns a plot arguments dictionary for the point x,y of the grid grid_object; if None: default pt_style is used for all points
                color_map:     standard color map used for pt_style's
        """
        if draw_surrounding_box:
            self.draw_surrounding_box(ax,surrounding_box_args)
        if draw_grid_lines:
            self.draw_grid_lines(ax,grid_line_args)
        if draw_points:
            self.draw_points(ax,default_pt_args,pt_style_callback,color_map,show_colorbar,ax_colorbar,colorbar_range,colorbar_label)
        if annotate:
            self.annotate(ax,self.title)
    
    def annotate(self,ax,text,xytext=(5,5),textcoords='offset pixels',annotation_clip=None,text_args={}):
        """ plot an annotation on ax at the position at which the spectrum was taken
        Arguments:
            ax (mpl axes): axes to plot into
            text (string): text to show
            xytext (float,float): position of the text, default: pixel relative to the point plotted by plot_position
            textcoords (string): coordinates used to specify the position of the text.
            args (dict): dictionary with additional arguments passed to annotate or text
        for further information about additional arguments see:
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.annotate.html and
        https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text
        """
        rad = np.radians(self.angle)
        xy = (
            self.center_x - np.cos(rad)*self.size_x/2-np.sin(rad)*self.size_y/2,
            self.center_y + np.cos(rad)*self.size_y/2-np.sin(rad)*self.size_x/2
        )
        ax.annotate(text,xy,xytext,'data',textcoords,annotation_clip,rotation=self.angle,**text_args)
        
    def extract_map(self,extractor_callback):
        """ Extract a map where the value for each point is calculated from the spectrum at that point by extractor_callback
        Arguments:
            extractor_callback (function(spectrum,x,y)):
                Arguments:
                    spectrum:   Nan_BiasSpectrum object of the concerned point
                    x:          x index of the concerned point in the grid
                    y:          y index of the concerned point in the grid
                Returns:
                    float:      Value to put in the map at the position x,y
        Returns:
            Map object with same size and position as the grid containing the values extracted by extractor_callback
        """
        res = np.zeros((self.pts_x,self.pts_y))
        for x in range(self.pts_x):
            for y in range(self.pts_y):
                res[x][y] = extractor_callback(self.get_spectrum(x,y),x,y)
        return Map(data=res,x_label='x',x_unit='m',y_label='y',y_unit='m',center_position=(self.center_x,self.center_y),size=(self.size_x,self.size_y),angle=self.angle)

    @staticmethod
    def const_bias_extractor(spec,x,y,signal,bias):
        s = spec.get_curve(signal)
        return s.evaluate(bias)

    def extract_const_bias_map(self,signal,bias):
        map = self.extract_map(
            lambda s,x,y: self.const_bias_extractor(s,x,y,signal,bias))
        map.set_data_label(signal)
        map.set_title(f'{self.title} at {bias}V')
        return map

    @staticmethod
    def topo_extractor(spec,x,y):
        return spec.get('Z (m)')

    def extract_topo_map(self):
        map = self.extract_map(self.topo_extractor)
        map.set_data_label('Topography')
        map.set_data_unit('m')
        map.set_colormap('afmhot')
        return map

    def map_spectra_along(self,signal='dIdV',scan_dir='avg',index=None,axis='x',other_axis_index=0,dig_dIdV_Gauss_kernel_size=3,relative_distance=False):
        """ Map the Spectra along a line cut along x or y.
        Arguments:
            signal (string): 'dIdV' for the numerical derivative of the current, or the name of the signal e.g. Input 3 (V) or Current (A)
            axis (string):  'x' for a cut along the x axis at y = other_axis_index
                            'y' for a cut along the y axis at x = other_axis_index
            other_axis_index (int): index along the other axis at which to cut
            dig_dIdV_Gauss_kernel_size (int): Gaussian smoothing of current before numerical derivation, passed on to BiasSpectrum get_dIdV. ignored when dIdV is read from lock-in signal
            relative_distance (bool): give the distance on the spatial axis of the map relative to the first point of the line cut
        Returns:
            (Map,pt_start,pt_end)
                Map containing the data of signal. Along the first axis, the spacial coordiante, along the second axis the bias voltage
                pt_start (float,float): position of the start of the cut
                pt_stop (float,float): position of the end of the cut 
        """
        spacial_points = self.pts_x if axis == 'x' else self.pts_y
        s0 = self.get_spectrum(0,0)
        spectrum_points = int(s0.get('num_pt'))
        allowed_signals = list(s0.get_available_signals())+ ['dIdV']
        allowed_signals.remove('params')
        allowed_signals.remove('topo')
        allowed_signals.remove('Bias calc (V)')
        if not signal in allowed_signals:
            raise ValueError(f"The Grid Spectrum '{self.filename}' does not contain a signal '{signal}'!")
        bias_range = np.sort([float(s0.get('V_start')),float(s0.get('V_stop'))])
        map_data = np.zeros((spacial_points,spectrum_points))

        x = range(spacial_points) if axis == 'x' else (other_axis_index*np.ones(spacial_points)).astype(int)
        y = range(spacial_points) if axis == 'y' else (other_axis_index*np.ones(spacial_points)).astype(int)

        first_spec = self.get_spectrum(np.min(x),np.min(y))
        last_spec = self.get_spectrum(np.max(x),np.max(y))
        param = 0 if axis == 'x' else 1
        spacial_range = (first_spec.get_position()[param],last_spec.get_position()[param])
        if relative_distance: 
            l = last_spec.get_position()
            f = first_spec.get_position()
            spacial_range = (0,np.sqrt((l[0]-f[0])**2+(l[1]-f[1])**2))

        for i in range(spacial_points):
            if signal == 'dIdV':
                map_data[i] = self.get_spectrum(x[i],y[i]).get_dIdV(lock_in_signal=None,scan_dir=scan_dir,index=index,dig_dIdV_Gauss_kernel_size=dig_dIdV_Gauss_kernel_size).get_y()
            else:
                map_data[i] = self.get_spectrum(x[i],y[i]).get_curve(signal,scan_dir,index).get_y()

        data_range = (spacial_range,bias_range)
        angle = 0 if axis == 'x' else 90 # todo exchange x and y label and units if axis is y.
        x_label = axis      if axis == 'x' else 'V Bias'
        x_unit = 'm'        if axis == 'x' else 'V'
        y_label = 'V Bias'  if axis == 'x' else axis
        y_unit = 'V'        if axis == 'x' else 'm'
        map = Map(map_data,x_label=x_label,x_unit=x_unit,y_label=y_label,y_unit=y_unit,data_label=signal,data_range=data_range,angle=angle)
        pt_start = first_spec.get_position()
        pt_end = last_spec.get_position()
        return (map,pt_start,pt_end)

    def map_linespectra_of(self,spacial_range,signal='dIdV',scan_dir='avg',index=None,axis_label="Label",axis_unit="Unit",other_axis_index=0,dig_dIdV_Gauss_kernel_size=3):
        """ Map the Spectra along a line cut along given spacial range=(,)
        Arguments:
            signal (string): 'dIdV' for the numerical derivative of the current, or the name of the signal e.g. Input 3 (V) or Current (A)
            axis (string):  'x' for a cut along the x axis at y = other_axis_index
                            'y' for a cut along the y axis at x = other_axis_index
            other_axis_index (int): index along the other axis at which to cut
            dig_dIdV_Gauss_kernel_size (int): Gaussian smoothing of current before numerical derivation, passed on to BiasSpectrum get_dIdV. ignored when dIdV is read from lock-in signal
            relative_distance (bool): give the distance on the spatial axis of the map relative to the first point of the line cut
        Returns:
            (Map,pt_start,pt_end)
                Map containing the data of signal. Along the first axis, the spacial coordiante, along the second axis the bias voltage
                pt_start (float,float): position of the start of the cut
                pt_stop (float,float): position of the end of the cut 
        """
        spacial_points = self.number_of_valid_spectra()
        s0 = self.get_spectrum(0,0)
        spectrum_points = int(s0.get('num_pt'))
        allowed_signals = list(s0.get_available_signals())+ ['dIdV']
        allowed_signals.remove('params')
        allowed_signals.remove('topo')
        allowed_signals.remove('Bias calc (V)')
        if not signal in allowed_signals:
            raise ValueError(f"The Grid Spectrum '{self.filename}' does not contain a signal '{signal}'!")
        bias_range = np.sort([float(s0.get('V_start')),float(s0.get('V_stop'))])
        map_data = np.zeros((spacial_points,spectrum_points))

        x = range(spacial_points) 
        y = (other_axis_index*np.ones(spacial_points)).astype(int)

        first_spec = self.get_spectrum(np.min(x),np.min(y))
        last_spec = self.get_spectrum(np.max(x),np.max(y))

        for i in range(spacial_points):
            if signal == 'dIdV':
                map_data[i] = self.get_spectrum(x[i],y[i]).get_dIdV(lock_in_signal=None,scan_dir=scan_dir,index=index,dig_dIdV_Gauss_kernel_size=dig_dIdV_Gauss_kernel_size).get_y()
            else:
                map_data[i] = self.get_spectrum(x[i],y[i]).get_curve(signal,scan_dir,index).get_y()

        data_range = (spacial_range,bias_range)
        angle = 0 
        x_label = axis_label  
        x_unit = axis_unit 
        y_label = 'V Bias'
        y_unit = 'V' 
        map = Map(map_data,x_label=x_label,x_unit=x_unit,y_label=y_label,y_unit=y_unit,data_label=signal,data_range=data_range,angle=angle)
        pt_start = first_spec.get_position()
        pt_end = last_spec.get_position()
        return (map,pt_start,pt_end)


class NanScan(NanonisMeasurement):
    """Class for Nanonis Scans.
        
    Properties:
        filename (string):      full filepath to the data file, including the filename 
        timestamp (datetime):   timestamp of the measurement
        title (string):         title of the measurement used for labeling the data in plots
        pixels ((int,int)):     number of pixels in x and y direction
        size_x (float):         size of the scan along the x (fast) axis in m
        size_y (float):         size of the scan along the y (slow) axis in m

        optional:
        sample (string):        name of the sample on which the file was measured
        T (float):              measurement temperature in K
        area (int or string):   ID of the area where the measurement was performed. The area changes when the sample moves or the tip is heavily altered, such that previous measurements are not comparable to this one anymore. 
    """

    def __init__(self,title,filename,sample=None, T=None, area=None):
        self.data = nap.read.Scan(filename)
        self.origin = 'upper' if self.data.header['scan_dir'] == 'down' else 'lower'
        self.angle = -float(self.data.header['scan_angle']) # nanonis angle is clockwise...
        try:
            sf= self.data.header['scan>scanfield'].split(';')
            self.center_x = float(sf[0])
            self.center_y = float(sf[1])
            self.size_x = float(sf[2])
            self.size_y = float(sf[3])
        except:
            self.center_x = self.data.header['scan_offset'][0]
            self.center_y = self.data.header['scan_offset'][1]
            self.size_x = self.data.header['scan_range'][0]
            self.size_y = self.data.header['scan_range'][1]

        self.pixels = self.data.header['scan_pixels']
        ts = datetime.strptime(self.data.header['rec_date']+self.data.header['rec_time'],'%d.%m.%Y%H:%M:%S')
        super().__init__(title,'Nan_Scan',filename,sample,ts,T,area)

    def scale_real_space(self,factor_x,factor_y):
        """ Scale the real space coordinates of the scan (position & size)
        Arguments:
            factor_x (float): scaling factor along the x-axis
            factor_y (float): scaling factor along the y-axis
        Returns:
            Nothing
        """
        self.center_x *=factor_x
        self.center_y *=factor_y
        self.size_x *=factor_x
        self.size_y *=factor_y
    def move_to(self,center_position):
        """ Move the center of the scan to the given position.
        Arguments:
            center_position (float,float): new center position, (x,y) 
        """
        self.center_x = center_position[0]
        self.center_y = center_position[1]
    def move_by(self,displacement):
        """ Displace the scan by the given vector.
        Arguments:
            displacement (float,float): displacement vector
        """
        self.center_x += displacement[0]
        self.center_y += displacement[1]
    def rotate(self,angle):
        """ Rotate the map by the given angle.
        Arugments:
            angle (float): angle to rotate the map in degree
        """
        self.angle += angle
    def get_map(self,signal,scan_direction='fwd',colormap='afmhot'):
        """ Get a map object for the desired signal
        Arguments:
            signal (string): the signal to return. e.g. 'Z', 'Current','Input 3'
            scan_direction (string): which scan direction to return 'fwd' for forward, 'bwd' for backward scan
            colormap: colormap of the map
        Returns:
            Map object containing the desired signal
        """
        dir = {'fwd': 'forward','bwd':'backward'}
        d = self.data.signals[signal][dir[scan_direction]]
        d = np.transpose(d)
        if self.origin == 'upper':
            d = np.flip(d,axis=1) # invert order of y coordinate to get the origin to the lower left corner
        return Map(d,
                    x_label='x',x_unit='m',
                    y_label='y',y_unit='m',
                    data_label=f'{signal} ({dir[scan_direction]})',
                    center_position=(self.center_x,self.center_y),
                    size=(self.size_x,self.size_y),
                    angle=self.angle,
                    colormap=colormap)

    def setup_plot(self,ax,signal,scan_direction='fwd',colormap='afmhot'):
        """Prepare the plot by setting axes labels according to the data
        
        Arguments:
            ax (matplotlib.axes):   axes to plot in
            signal (string): the signal to return. e.g. 'Z', 'Current','Input 3'
            scan_direction (string): which scan direction to return 'fwd' for forward, 'bwd' for backward scan
            colormap: colormap of the map
        """
        m = self.get_map(signal,scan_direction=scan_direction,
                        colormap=colormap)
        m.setup_plot(ax)

    def annotate(self,ax,text,xytext=(5,5),textcoords='offset pixels',annotation_clip=None,text_args={}):
        """ plot an annotation on ax at the position at which the spectrum was taken
        Arguments:
            ax (mpl axes): axes to plot into
            text (string): text to show
            xytext (float,float): position of the text, default: pixel relative to the point plotted by plot_position
            textcoords (string): coordinates used to specify the position of the text.
            args (dict): dictionary with additional arguments passed to annotate or text
        for further information about additional arguments see:
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.annotate.html and
        https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text
        """
        rad = np.radians(self.angle)
        xy = (
            self.center_x - np.cos(rad)*self.size_x/2-np.sin(rad)*self.size_y/2,
            self.center_y + np.cos(rad)*self.size_y/2-np.sin(rad)*self.size_x/2
        )
        ax.annotate(text,xy,xytext,'data',textcoords,annotation_clip,rotation=self.angle,**text_args)

    def plot(self,ax,signal,scan_direction='fwd',colormap='afmhot', ax_colorbar=None,imshow_args=None,annotate=False):
        """Plot the scan in a matplotlib axes.

        Arguments:
            ax (matplotlib.axes):       axes to plot into
            signal (string):            the signal to return. e.g. 'Z', 'Current','Input 3'
            scan_direction (string):    which scan direction to return 'fwd' for forward, 'bwd' for backward scan
            colormap:                   colormap of the map
            ax_colorbar: (mpl.axes)     axes to plot the colorscale into
            imshow_args (dict):         arguments passed on to matplotlib.pyplot.imshow
        """
        m = self.get_map(signal,scan_direction=scan_direction,
                        colormap=colormap)
        m.plot(ax,ax_colorbar,imshow_args)
        if annotate:
            self.annotate(ax,self.title)
    
    def plot_standalone(self,signal,scan_direction='fwd',colormap='afmhot', show_colorbar=True,imshow_args=None):
        """Convenience function to plot the scan in a new figure.

        Arguments:
            signal (string):            the signal to return. e.g. 'Z', 'Current','Input 3'
            scan_direction (string):    which scan direction to return 'fwd' for forward, 'bwd' for backward scan
            colormap:                   colormap of the map
            ax_colorbar: (mpl.axes)     axes to plot the colorscale into
            imshow_args (dict):         arguments passed on to matplotlib.pyplot.imshow
        """
        gs = GridSpec(1,20)
        fig=plt.figure()
        if show_colorbar:
            ax = fig.add_subplot(gs[0,:18])
            ax_colorbar = fig.add_subplot(gs[0,19])
        else:
            ax = fig.add_subplot(gs[:,:])
            ax_colorbar = None
        m = self.get_map(signal,scan_direction=scan_direction,
                        colormap=colormap)
        m.setup_plot(ax)
        m.plot(ax,ax_colorbar,imshow_args)
        plt.show()

    def get_available_signals(self):
        """Get a list of all available signals in the data file.
        
        Returns:
            list of strings: containing the names of the available signals"""
        return self.data.signals.keys()