import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


class Data():
    """
    A parser for known data files in txt format. 
    """

    def __init__(self, filename):
        """
        A parser for data files in txt format.
        
        :param filename: the name of the file to be parsed 
        """

        self._filename = filename
        self._define_type()
        
        self.data = self._parse()


    def _define_type(self):
        """
        Determine the file type for parsing.

        :return: array of names of cols in file
        """
        if 'UHECR' in self._filename:
            self._filetype = 'uhecr'
            self._filelayout = ['year', 'day', 'incidence angle',
                      'energy', 'ra', 'dec',
                      'glon', 'glat']

        elif 'agn' in self._filename:
            self._filetype = 'agn'
            self._filelayout = ['name', 'glon', 'glat', 'D']
        
        else:
            print ('File layout not recognised')
            self._filetype = None 
            self._filelayout = None


    
    def _parse(self):
        """
        Parse the data form the object's file.
        
        :return: arrays for each column in the data file
        """

        output = pd.read_csv(self._filename, comment = '#',
                             delim_whitespace = True,
                             names = self._filelayout)

        output_dict = output.to_dict()
        
        return output_dict


    def get_len(self):
        """
        Get the length of the data set.

        :return: the length of the data set
        """

        if self._filetype == 'uhecr':
            n = len(self.data['year'])

        elif self._filetype == 'agn':
            n = len(self.data['name'])

        return n

    
    def get_coordinates(self):
        """
        Get the galactic coordinates from self.data
        and return them as astropy SkyCoord
        
        :return: astropy.coordinates.SkyCoord
        """

        glon = np.array( list(self.data['glon'].values()) )
        glat = np.array( list(self.data['glat'].values()) )

        return SkyCoord(l = glon * u.degree, b = glat * u.degree, frame = 'galactic')

    
    
    def get_by_name(self, name):
        """
        Get data entries by name.

        :param name: name of the data as in self._filelayout
        :return: an array of data entries
        """

        try:
            selected_data = np.array( list(self.data[name].values()) )

        except ValueError:
            print ('No data of type', name)
            selected_data = []

        return selected_data
