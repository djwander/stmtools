from os import times
from pathlib import *
import numpy as np
from datetime import datetime,timedelta, timezone
import pandas as pd

class Sionludi():
    """Class to load data from the Sionludi .xls histoy files."""

    get_shortcuts = {   'Dilution_Temp':'T3@JT2',
                        'STM_Temp':'T1@JT2',
                        'P1': 'J4',
                        'P2': 'J5',
                        }

    def __init__(self,path):
        """Create a Sionludi object.

        Arguments:
            path (string):  path to the folder containing the data files
        """
        self.path=path
        self._loaded_files = []
        self.data = pd.DataFrame()

    def _parse_filename(self,timestamp):
        """Generate the filename containing the data for the specified day of timestamp

        Arguments:
            timestamp (datetime): timestamp of the day
        """
        return "data-sionludi_Ch_Proche3_"+datetime.strftime(timestamp,"%Y-%m-%d")+".xls"

    def _make_file_register(self, timestamps):
        """Create a list of files corresponding to the dates covered by the timestamps.
        Arguments:
            timestamps (list of datetime): timestamps for which the corresponding filenames are generated
        """
        dates = set(ts.date() for ts in timestamps)
        return [self._parse_filename(d) for d in dates]

    @staticmethod
    def _ts_to_s_since_epoch(timestamp):
        return (timestamp-datetime(1970,1,1)).total_seconds()

    def load_data(self,timestamps):
        """Load the files of all dates corresponding to the given timestamps

        Arguments:
            timestamps (list of datetime): timestamps for which the data files are loaded
        """
        files = self._make_file_register(timestamps)
        self.load_files(files)

    def load_files(self,files):
        """Load the data from the given files into memory
        
        Arguments:
            files (list of strings): list of filenames to be loaded
            .time includes all timestamps as datetime
            .column_names includes all columns with available data
            .values includes all values from the columns
        
        """
        folder = Path(self.path)

        for f in files:
            if not f in self._loaded_files:
                filepath = folder / f
                if not filepath.is_file():
                    raise ValueError(f'Problem when loading Sionludi data. Could not find "{f}" in "{folder}"!')
                data = pd.read_csv(filepath,decimal=',',delimiter='\t')
                data['Date'] = pd.to_datetime(data['Date'],format="%d/%m/%Y %H:%M:%S")
                data['t'] = data.apply(lambda row: self._ts_to_s_since_epoch(row['Date']),axis=1)
                data.dropna(inplace=True)
                self.data = self.data.append(data,ignore_index=True)
                self.data = self.data.sort_values(by=['t']).reset_index(drop=True)
                self._loaded_files.append(f)

    def get_value(self,timestamp,column):
        """Get single value.

        Arguments:
            timestamp (datetime):   timestamp at which to load the value
            column (string):        name of the signal to load the value from
        
        Returns:
            float: value
        """
        self.load_data([timestamp])
        closest_entry_index = np.argmin((np.abs(self.data['t']-self._ts_to_s_since_epoch(timestamp))).values)
        return self.data[column].values[closest_entry_index]
    
    @staticmethod
    def all_days_between(ts_from,ts_to):
        """Generate a list with timestamps for all days in the range [ts_from,ts_to]
        Arguments:
            ts_from (datetime):     timestamp of the day from which to start
            ts_to (datetime):       timestamp of the day at which to stop
        Returns:
            [datetimes]:            list with timestamps of all days in the range
        """
        timestamps = [ts_from]
        ts = ts_from
        dt = timedelta(days=1)
        while ts.date() < ts_to.date():
            ts += dt
            timestamps.append(ts)
        return timestamps

    def get_values_range(self,ts_from,ts_to,column):
        """Get all values in a certain time range.

        Arguments:
            ts_from (datetime):     timestamp from which to load the values
            ts_to (datetime):       timestamp to which to load the values
            column (string):        name of the signal to load the values from

        Returns:
            np.array of floats: values within the time range
        """
        self.load_data(self.all_days_between(ts_from,ts_to))
        if (ts_to-ts_from).total_seconds() < 0:
            raise ValueError(f'ts_from ({ts_from}) is before ts_to ({ts_to})!')
        start_index = np.argmax((np.sign(self.data['t']-self._ts_to_s_since_epoch(ts_from))).values)
        stop_index =  np.argmax((np.sign(self.data['t']-self._ts_to_s_since_epoch(ts_to))).values)-1
        return np.array(self.data[column].values[start_index:stop_index])
