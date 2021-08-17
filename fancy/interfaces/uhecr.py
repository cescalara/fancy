import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from datetime import date, timedelta
import h5py
from scipy.optimize import root
from tqdm import tqdm as progress_bar

from .stan import coord_to_uv, uv_to_coord
from ..detector.detector import Detector
from ..plotting import AllSkyMap
from .utils import get_nucleartable

# importing crpropa, need to append system path
import sys
sys.path.append("/opt/CRPropa3/lib/python3.8/site-packages")
import crpropa

__all__ = ['Uhecr']


class Uhecr():
    """
    Stores the data and parameters for UHECRs
    """
    def __init__(self):
        """
        Initialise empty container.
        """

        self.properties = None
        self.source_labels = None

        self.nuc_table = get_nucleartable()

    def _get_angerr(self):
        '''Get angular reconstruction uncertainty from label'''

        if self.label == "TA2015":
            from fancy.detector.TA2015 import sig_omega
        elif self.label == "auger2014":
            from fancy.detector.auger2014 import sig_omega
        elif self.label == "auger2010":
            from fancy.detector.auger2010 import sig_omega
        else:
            raise Exception("Undefined detector type!")

        return np.deg2rad(sig_omega)
        

    def from_data_file(self, filename, label,  ptype="p", exp_factor = 1.):
        """
        Define UHECR from data file of original information.
        
        Handles calculation of observation periods and 
        effective areas assuming the UHECR are detected 
        by the Pierre Auger Observatory or TA.
        
        :param filename: name of the data file
        :param label: reference label for the UHECR data set 
        """

        self.label = label

        with h5py.File(filename, 'r+') as f:

            data = f[self.label]

            self.year = data['year'][()]
            self.day = data['day'][()]
            self.zenith_angle = np.deg2rad(data['theta'][()])
            self.energy = data['energy'][()]
            self.N = len(self.energy)
            glon = data['glon'][()]
            glat = data['glat'][()]
            self.coord = self.get_coordinates(glon, glat)

            self.unit_vector = coord_to_uv(self.coord)
            self.period = self._find_period()
            self.A = self._find_area(exp_factor)

        self.ptype = ptype

    def _get_properties(self):
        """
        Convenience function to pack object into dict.
        """

        self.properties = {}
        self.properties['label'] = self.label
        self.properties['N'] = self.N
        self.properties['unit_vector'] = self.unit_vector
        self.properties['energy'] = self.energy
        self.properties['A'] = self.A
        self.properties['zenith_angle'] = self.zenith_angle
        self.properties["ptype"] = self.ptype

        # Only if simulated UHECRs
        if isinstance(self.source_labels, (list, np.ndarray)):
            self.properties['source_labels'] = self.source_labels

    def from_properties(self, uhecr_properties):
        """
        Define UHECR from properties dict.
            
        :param uhecr_properties: dict containing UHECR properties.
        :param label: identifier
        """

        self.label = uhecr_properties['label']

        # Read from input dict
        self.N = uhecr_properties['N']
        self.unit_vector = uhecr_properties['unit_vector']
        self.energy = uhecr_properties['energy']
        self.zenith_angle = uhecr_properties['zenith_angle']
        self.A = uhecr_properties['A']
        self.ptype = uhecr_properties['ptype']

        # Only if simulated UHECRs
        try:
            self.source_labels = uhecr_properties['source_labels']
        except:
            pass

        # Get SkyCoord from unit_vector
        self.coord = uv_to_coord(self.unit_vector)

    def plot(self, skymap, size=2):
        """
        Plot the Uhecr instance on a skymap.

        Called by Data.show()
      
        :param skymap: the AllSkyMap
        :param size: tissot radius
        :param source_labels: source labels (int)
        """

        lons = self.coord.galactic.l.deg
        lats = self.coord.galactic.b.deg

        alpha_level = 0.7

        # If source labels are provided, plot with colour
        # indicating the source label.
        if isinstance(self.source_labels, (list, np.ndarray)):

            Nc = max(self.source_labels)

            # Use a continuous cmap
            cmap = plt.cm.get_cmap('plasma', Nc)

            write_label = True

            for lon, lat, lab in np.nditer([lons, lats, self.source_labels]):
                color = cmap(lab)
                if write_label:
                    skymap.tissot(lon,
                                  lat,
                                  size,
                                  npts=30,
                                  facecolor=color,
                                  alpha=0.5,
                                  label=self.label)
                    write_label = False
                else:
                    skymap.tissot(lon,
                                  lat,
                                  size,
                                  npts=30,
                                  facecolor=color,
                                  alpha=0.5)

        # Otherwise, use the cmap to show the UHECR energy.
        else:

            # use colormap for energy
            norm_E = matplotlib.colors.Normalize(min(self.energy),
                                                 max(self.energy))
            cmap = plt.cm.get_cmap('viridis', len(self.energy))

            write_label = True
            for E, lon, lat in np.nditer([self.energy, lons, lats]):

                color = cmap(norm_E(E))

                if write_label:
                    skymap.tissot(lon,
                                  lat,
                                  size,
                                  30,
                                  facecolor=color,
                                  alpha=alpha_level,
                                  label=self.label)
                    write_label = False
                else:
                    skymap.tissot(lon,
                                  lat,
                                  size,
                                  30,
                                  facecolor=color,
                                  alpha=alpha_level)

    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with 
        file_handle.create_dataset()
        
        :param file_handle: file handle
        """

        self._get_properties()

        for key, value in self.properties.items():
            file_handle.create_dataset(key, data=value)

    def _find_area(self, exp_factor):
        """
        Find the effective area of the observatory at 
        the time of detection.

        Possible areas are calculated from the exposure reported
        in Abreu et al. (2010) or Collaboration et al. 2014.
        """

        if self.label == 'auger2010':
            from ..detector.auger2010 import A1, A2, A3
            possible_areas = [A1, A2, A3]
            area = [possible_areas[i - 1]* exp_factor  for i in self.period]

        elif self.label == 'auger2014':
            from ..detector.auger2014 import A1, A2, A3, A4, A1_incl, A2_incl, A3_incl, A4_incl
            possible_areas_vert = [A1, A2, A3, A4]
            possible_areas_incl = [A1_incl, A2_incl, A3_incl, A4_incl]

            # find area depending on period and incl
            area = []
            for i, p in enumerate(self.period):
                if self.zenith_angle[i] <= 60:
                    area.append(possible_areas_vert[p - 1]* exp_factor )
                if self.zenith_angle[i] > 60:
                    area.append(possible_areas_incl[p - 1]* exp_factor )

        elif self.label == 'TA2015':
            from ..detector.TA2015 import A1, A2
            possible_areas = [A1, A2]
            area = [possible_areas[i - 1] * exp_factor for i in self.period]

        else:
            print('Error: effective areas and periods not defined')

        return area

    def _find_period(self):
        """
        For a given year or day, find UHECR period based on dates
        in table 1 in Abreu et al. (2010) or in Collaboration et al. 2014.
        """

        period = []
        if self.label == "auger2014":
            from ..detector.auger2014 import (period_1_start, period_1_end,
                                            period_2_start, period_2_end,
                                            period_3_start, period_3_end,
                                            period_4_start, period_4_end)

            # check dates
            for y, d in np.nditer([self.year, self.day]):
                d = int(d)
                test_date = date(y, 1, 1) + timedelta(d)

                if period_1_start <= test_date <= period_1_end:
                    period.append(1)
                elif period_2_start <= test_date <= period_2_end:
                    period.append(2)
                elif period_3_start <= test_date <= period_3_end:
                    period.append(3)
                elif test_date >= period_3_end:
                    period.append(4)
                else:
                    print('Error: cannot determine period for year', y, 'and day',
                        d)

        elif self.label == "TA2015":
            from ..detector.TA2015 import (period_1_start, period_1_end,
                                            period_2_start, period_2_end)
            for y, d in np.nditer([self.year, self.day]):
                d = int(d)
                test_date = date(y, 1, 1) + timedelta(d)

                if period_1_start <= test_date <= period_1_end:
                    period.append(1)
                elif period_2_start <= test_date <= period_2_end:
                    period.append(2)
                elif test_date >= period_2_end:
                    period.append(2)
                else:
                    print('Error: cannot determine period for year', y, 'and day',
                        d)    

        return period

    def select_period(self, period):
        """
        Select certain periods for analysis, other periods will be discarded. 
        """

        # find selected periods
        if len(period) == 1:
            selection = np.where(np.asarray(self.period) == period[0])
        if len(period) == 2:
            selection = np.concatenate([
                np.where(np.asarray(self.period) == period[0]),
                np.where(np.asarray(self.period) == period[1])
            ],
                                       axis=1)

        # keep things as lists
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection]

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]

    def select_energy(self, Eth):
        """
        Select out only UHECRs above a certain energy.
        """

        selection = np.where(np.asarray(self.energy) >= Eth)
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection]

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]


    def get_coordinates(self, glon, glat, D=None):
        """
        Convert glon and glat to astropy SkyCoord
        Add distance if possible (allows conversion to cartesian coords)
            
        :return: astropy.coordinates.SkyCoord
        """

        if D:
            return SkyCoord(l=glon * u.degree,
                            b=glat * u.degree,
                            frame='galactic',
                            distance=D * u.mpc)
        else:
            return SkyCoord(l=glon * u.degree, b=glat * u.degree, frame='galactic')

    def coord_to_vector3d(self):
        '''Convert from SkyCoord array to Vector3d list'''
        uhecr_vector3d = []

        # due to how SkyCoord defines coordinates,
        # lon_vector3d = pi - lon_skycoord
        # lat_vector3d = pi/2 - lat_skycoord
        for coord in self.coord:
            v = crpropa.Vector3d()
            v.setRThetaPhi(1, np.pi / 2. - coord.galactic.b.rad, np.pi - coord.galactic.l.rad)
            uhecr_vector3d.append(v)
        return uhecr_vector3d

    # def get_AZ(self, nuc="p"):
    #     '''Return (A, Z) of the given nuclear element. Note that 'p'=='H'.'''
    #     return self.nuc_table[nuc]

    def _prepare_crpropasim(self, particle_type="p", model_name="JF12", seed=691342):
        '''Set up CRPropa simulation with some magnetic field model'''
        sim = crpropa.ModuleList()

        # setup magnetic field
        if model_name == "JF12":  # JanssonFarrar2012
            gmf = crpropa.JF12Field()
            gmf.randomStriated(seed)
            gmf.randomTurbulent(seed)

        elif model_name == "PT11":  # Pshirkov2011 
            gmf = crpropa.PT11Field()
            gmf.setUseASS(True)  # Axisymmetric

        else:  # default with JF12
            gmf = crpropa.JF12Field()
            gmf.randomStriated(seed)
            gmf.randomTurbulent(seed)

        # Propagation model, parameters: (B-field model, target error, min step, max step)
        sim.add(crpropa.PropagationCK(gmf, 1e-4, 0.1 * crpropa.parsec, 100 * crpropa.parsec))
        obs = crpropa.Observer()

        # observer at galactic boundary (20 kpc)
        obs.add(crpropa.ObserverSurface( crpropa.Sphere(crpropa.Vector3d(0), 20 * crpropa.kpc) ))
        sim.add(obs)

        # A, Z = self.get_AZ(nuc=particle_type)
        A, Z = self.nuc_table[particle_type]

        # composition
        pid = - crpropa.nucleusId(A, Z)

        # CRPropa random number generator
        crpropa_randgen = crpropa.Random() 

        # position of earth in galactic coordinates
        pos_earth = crpropa.Vector3d(-8.5, 0, 0) * crpropa.kpc

        return sim, pid, crpropa_randgen, pos_earth

    def plot_deflections(self, defl_arrdirs, rand_arrdirs):
        '''Plot deflections for particular UHECR dataset'''
        # check with basic mpl mollweide projection
        plt.figure(figsize=(12,7))
        ax = plt.subplot(111, projection = 'mollweide')

        # get lon and lat arrays for future reference
        # shift lons by 180. due to how its defined in mpl
        uhecr_lons = np.pi - self.coord.galactic.l.rad
        uhecr_lats = self.coord.galactic.b.rad

        ax.scatter(uhecr_lons, uhecr_lats, color="k", marker="+", s=10.0, alpha=1., label="True")
        for i in range(self.N):
            ax.scatter(defl_arrdirs[i, :, 0], defl_arrdirs[i, :, 1], color="b", alpha=0.05, s=4.0)
            ax.scatter(rand_arrdirs[i, :, 0], rand_arrdirs[i, :, 1], color="r", alpha=0.05, s=4.0)

        # TODO: add labels based on color 
        ax.grid()


    def eval_kappad(self, kappad_args):
        '''
        Evaluate spread parameter between arrival direction
        and deflected direction at galactic magnetic field.
        Returns kappa_d, deflected and undeflected arrival vectors
        '''

        particle_type, Nrand, gmf, plot = kappad_args

        # get angular reconstruction uncertainty
        ang_err = self._get_angerr()
        # evaluate vector3d object of arrival direction
        uhecr_vector3d = self.coord_to_vector3d()
        # prepare inputs / initializations for CRPropa simulation
        sim, pid, R, pos = self._prepare_crpropasim(particle_type, gmf)

        # arrival directions, mainly for plotting
        rand_arrdirs = np.zeros((self.N, Nrand, 2))
        defl_arrdirs = np.zeros((self.N, Nrand, 2))

        # evaluated kappa_d
        kappa_ds = np.zeros((self.N, Nrand))

        for i, arr_dir in enumerate(uhecr_vector3d):
            energy = self.energy[i] * crpropa.EeV
            for j in range(Nrand):
                rand_arrdir = R.randVectorAroundMean(arr_dir, ang_err)
                c = crpropa.Candidate(crpropa.ParticleState(pid, energy, pos, rand_arrdir))
                sim.run(c)

                defl_dir = c.current.getDirection()

                # append longitudes and latitudes
                # need to append np.pi / 2 - theta for latitude
                # also append the randomized arrival direction in lons and lats
                rand_arrdirs[i, j, :] = rand_arrdir.getPhi(), np.pi / 2. - rand_arrdir.getTheta()
                defl_arrdirs[i, j, :] = defl_dir.getPhi(), np.pi / 2. - defl_dir.getTheta()

                # evaluate dot product between arrival direction (randomized) and deflected vector
                # dot exists with Vector3d() objects
                cos_theta = rand_arrdir.dot(defl_dir)

                # use scipy.optimize.root to get kappa_d using dot product
                # P = 0.683 as per Soiaporn paper
                sol = root(fischer_int_eq_P, x0=100, args=(cos_theta, 0.683))
                # print(sol)   # check solution

                kappa_ds[i, j] = sol.x[0]

        if plot:
            self.plot_deflections(defl_arrdirs, rand_arrdirs)

        # evaluate mean kappa_d for each uhecr
        kappa_d_mean = np.mean(kappa_ds, axis=1)

        return kappa_d_mean, defl_arrdirs, rand_arrdirs


'''Integral of Fischer distribution used to evaluate kappa_d'''
def fischer_int(kappa, cos_thetaP):
    '''Integral of vMF function over all angles'''
    return (1. - np.exp(-kappa * (1 - cos_thetaP))) / (1. - np.exp(-2.*kappa))

def fischer_int_eq_P(kappa, cos_thetaP, P):
    '''Equation to find roots for'''
    return fischer_int(kappa, cos_thetaP) - P