"""Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)"""

import pickle as pickle

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from fancy.utils.package_data import (
    get_path_to_lens,
)

try:
    import crpropa
except:
    crpropa = None

import healpy

class GMFLensing:
    """Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)."""

    __lens_names = {"JF12": "JF12full_Gamale", "UF23":"UF23_all", "UF23Turb" : "UF23Turb_all"}  # noqa: RUF012

    def __init__(self, gmf_model: str = "JF12") -> None:
        """
        Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps).

        gmf_model: str, default JF12
            The desired GMF model for GMF lensing
        """
        if crpropa is None:
            raise ImportError("CRPropa must be installed to use this functionality.")

        self.gmf_model = gmf_model

        # read in GMF lens if we have GMF enabled
        if gmf_model in list(self.__lens_names.keys()):
            path_to_lens = str(get_path_to_lens(self.__lens_names[self.gmf_model]))
            self.gmf_lens = crpropa.MagneticLens(path_to_lens)
            self.disable_gmf = False
        elif gmf_model == "None":
            self.disable_gmf = True
        else:
            raise NotImplementedError(
                f"Lensing for GMF model {gmf_model} not yet implemented."
            )

    def apply_lens_with_particles(self, rigidities : np.ndarray, coordinates: SkyCoord) -> SkyCoord:
        """
        Apply GMF lensing by sampling & re-sampling. Returns same number of sampled events at earth as SkyCoord objects.

        rigidities: np.ndarray
            rigidities from particle samples in EV
        coordinates: astropy.coordinates.SkyCoord
            arrival directions of samples in SkyCoord
        """
        # now GMF lensing
        particle_map = crpropa.ParticleMapsContainer()
        Nsamples = coordinates.shape[0]

        for i in range(Nsamples):
            # make coordinate system consistent
            coord_gb_xyz = -1 * coordinates[i].cartesian.xyz.value
            vector3d_gb = crpropa.Vector3d(*coord_gb_xyz)

            # adding rigidities instead of energy
            particle_map.addParticle(
                crpropa.nucleusId(1, 1), rigidities[i] * crpropa.EeV, vector3d_gb
            )

        # lens
        if not self.disable_gmf:
            particle_map.applyLens(self.gmf_lens)

        # sample back same number of particles at earth
        _, _, glon_earth, glat_earth = particle_map.getRandomParticles(int(Nsamples))

        return SkyCoord(
            glon_earth * u.rad,
            glat_earth * u.rad,
            frame="galactic",
            representation_type="unitspherical",
        )

    def apply_lens_to_map(self, weighted_map, R: float):
        """
        Apply GMF lensing from weighted healpy map

         weighted_map: 
            map of normalised counts that represent an event distribution at each coordinate. **must be of Pixelisation order 6 (NPIX = 49152) following CRPropa conventions**
         R: float
            rigidity in EV
        """

        # make sure dimensionality is of order 6
        if len(weighted_map) != 49152:
            raise ValueError(
                "Dimension of weighted map must be of order 6 (NPIX = 49152)!"
            )

        # also make sure weighted map is normalised
        assert (
            np.sum(weighted_map) < 1.01 and np.sum(weighted_map) > 0.99
        ), f"sum of unlensed weights = {np.sum(weighted_map)} != 1"

        # create maps container and add weights to it
        particles = crpropa.ParticleMapsContainer()
        particles.addWeights(R * crpropa.EeV, weighted_map)

        # apply lensing
        if not self.disable_gmf:
            particles.applyLens(R * crpropa.EeV, self.gmf_lens)

        # obtain the lensed weights
        lensed_weighted_map = particles.getWeights(
            crpropa.nucleusId(1, 1), R * crpropa.EeV
        )
        if np.any(np.isnan(lensed_weighted_map)):
            print(R, weighted_map[weighted_map > 0])
            print(len(np.isnan(lensed_weighted_map) == True))
            raise ValueError("NaN values in lensed weighted map!")

        assert (
            np.sum(lensed_weighted_map) < 1.01 and np.sum(lensed_weighted_map) > 0.99
        ), f"sum of lensed weights = {np.sum(lensed_weighted_map)} != 1"

        return lensed_weighted_map
